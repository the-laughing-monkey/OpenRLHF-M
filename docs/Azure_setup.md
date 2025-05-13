# Running OpenRLHF-M on Microsoft Azure

This guide shows how to deploy **OpenRLHF-M** on Azure using **Docker**, **Ray** for multi-node training, and a high-performance **NFS share (Azure NetApp Files or Azure Files NFS)** for datasets, model cache, and checkpoints.

The structure and naming conventions mirror the existing **RunPod** and **GCP** setup documents so that all three cloud guides feel familiar.

---

## Overview

We will use:

* **Azure CLI** (`az`) for all resource creation.
* **Azure NetApp Files** (ANF) **_or_** **Azure Files (NFS 4.1)** for a shared POSIX-compatible file system.  
  • ANF gives HPC-class performance but costs more.  
  • Azure Files NFS v4.1 is cheaper/slower but still adequate for many workloads.  
* **Azure Container Registry (ACR)** to host our custom Docker image.
* **GPU VMs** – e.g. `Standard_NC48ads_A100_v4` or `Standard_ND96amsr_A100_v4`.  
  We'll create one **head** and one or more **worker** nodes in the same **VNet & subnet**.
* **Ray** (started inside the containers) for distributed execution.
* **Weights & Biases** (optional) for experiment tracking.

> If you prefer full HPC automation, jump to **Option 2 – Azure CycleCloud** or **Option 3 – AKS + NetApp Files** at the end of this document.

---

## Prerequisites

1. An Azure subscription with **sufficient GPU quota** in your chosen region.  
   • List current GPU-related quotas/usage (replace `eastus` with your region):  
   ```bash
   # Quick view of all VM quotas and current usage
   az vm list-usage --location eastus --output table | grep -Ei 'gpu|ND|NC|NV'
   
   # Fine-grained quota API (preview but more explicit)
   az quota list --resource-type Standard_NC -l eastus --service-name Microsoft.Compute
   ```  
   If the "Current Value" is lower than what you need, open a request in the Azure Portal:  
   **Subscriptions ➜ Usage + quotas ➜ Request quota increase** (pick the GPU VM family, e.g. *Standard_NC* or *Standard_ND*).

   • **Finding GPU-enabled regions and VM sizes:**
     To list all Azure locations:
     ```bash
     az account list-locations --query "[].{Name:name, DisplayName:displayName}" -o table
     ```
     To see which VM sizes (including GPU) are available in a specific location (replace `eastus`):
     ```bash
     az vm list-sizes --location eastus --output table | grep -Ei 'gpu|NC|ND|NV|MI'
     ```
     Look for VM series like `NC`, `ND`, `NV`, `MI` which typically indicate GPU capabilities, and check the `numberOfGPUs` column.

2. **Azure CLI** (`az`) installed **and logged-in**.  
   Install (choose one):  
   ```bash
   # macOS (Homebrew):
   brew update && brew install azure-cli
   
   # Ubuntu / Debian:
   curl -sL https://aka.ms/InstallAzureCLIDeb | sudo bash
   
   # RHEL / CentOS / Fedora:
   curl -sL https://aka.ms/InstallAzureCLIRpm | sudo bash
   
   # Windows (PowerShell as Admin):
   msiexec /i https://aka.ms/installazurecliwindows
   ```
   Then sign in and select the working subscription:  
   ```bash
   az login                # opens browser
   az account set --subscription "<SUBSCRIPTION_ID>"
   ```

3. **Docker** (with Buildx) on your local workstation.
4. **Git** and a Unix-like shell (bash/PowerShell).
5. An **SSH key pair** for VM login (`ssh-keygen -t ed25519 -f ~/.ssh/azure_openrlhf`).
6. **(Optional)** a Weights & Biases account and `WANDB_API_KEY`.

---

## Variables

```bash
# ---------- Global ----------
export RG=openrlhf-rg
export LOCATION=eastus         # choose a GPU-enabled region
export VNET=openrlhf-vnet
export SUBNET=openrlhf-subnet
export ADDRESS_PREFIX=10.42.0.0/16
export SUBNET_PREFIX=10.42.0.0/24

# ---------- Storage ----------
export ANF_ACC=openrlhf-anf      # Only if using NetApp Files
export ANF_POOL=anfpool
export ANF_VOL=anfvol
export FILES_ACC=openrlhffiles   # Only if using Azure Files (NFS)
export NFS_PATH=/openrlhf-share  # Azure Files export OR ANF volume path
export MOUNT_POINT=/mnt/nfs

# ---------- Container ----------
export ACR_NAME=openrlhfacr
export IMAGE=openrlhf-m:latest

# ---------- Compute ----------
export HEAD_VM=openrlhf-head
export WORKER_VM=openrlhf-worker1
export VM_SIZE=Standard_NC48ads_A100_v4   # 4×A100-80GB
export ADMIN_USER=azureuser
export SSH_KEY_PATH=~/.ssh/azure_openrlhf.pub
export WANDB_API_KEY=<YOUR_WANDB_API_KEY>
```

---

## 1  Create Resource Group, VNet & Subnet

```bash
az group create -n $RG -l $LOCATION

az network vnet create            \
  -g $RG -n $VNET                 \
  --address-prefix $ADDRESS_PREFIX \
  --subnet-name $SUBNET           \
  --subnet-prefix $SUBNET_PREFIX
```

---

## 2  Provision Shared NFS Storage

### Option A  Azure NetApp Files (High Performance)

```bash
# Enable ANF provider once per subscription
az provider register --namespace Microsoft.NetApp

# Create ANF account → capacity pool → volume
az netappfiles account create -g $RG -n $ANF_ACC -l $LOCATION

az netappfiles pool create -g $RG --account-name $ANF_ACC \
  -n $ANF_POOL --size 4 --qos-type Auto              # 4 TiB min

az netappfiles volume create -g $RG --account-name $ANF_ACC \
  --pool-name $ANF_POOL -n $ANF_VOL \
  -l $LOCATION --file-path ${ANF_VOL} \
  --usage-threshold 1024        \   # 1 TiB
  --vnet $VNET --subnet $SUBNET \
  --protocol-types NFSv4.1

# Grab mount IP
export NFS_IP=$(az netappfiles volume show -g $RG \
  --account-name $ANF_ACC --pool-name $ANF_POOL \
  -n $ANF_VOL --query mountIpAddresses[0] -o tsv)
echo "ANF mount IP: $NFS_IP"
```

### Option B  Azure Files (NFS 4.1)

```bash
# Enable NFS v4.1
az storage account create -g $RG -n $FILES_ACC -l $LOCATION \
  --sku Premium_ZRS --kind FileStorage --enable-large-file-share

az storage share-rm create --storage-account $FILES_ACC \
  -g $RG -n $NFS_PATH --enabled-protocol NFS \
  --root-squash NoRootSquash --quota 1024
# Get the storage account FQDN
export NFS_IP=$(az storage account show -n $FILES_ACC -g $RG \
  --query primaryEndpoints.file -o tsv | sed 's#https://##;s#/##')
echo "Azure Files NFS endpoint: $NFS_IP"
```

---

## 3  Build & Push Docker Image to ACR

```bash
# Create ACR
az acr create -g $RG -n $ACR_NAME --sku Standard
az acr login -n $ACR_NAME
export ACR_LOGIN_SERVER=$(az acr show -n $ACR_NAME \
                          --query loginServer -o tsv)

# Build image (same Dockerfile as GCP guide)
docker buildx build --platform linux/amd64 \
  -t ${ACR_LOGIN_SERVER}/${IMAGE} \
  -f dockerfile/gcp/Dockerfile .      # reuse existing Dockerfile
docker push ${ACR_LOGIN_SERVER}/${IMAGE}
```

---

## 4  Prepare Data & Models on the NFS Share

Create a **temporary prep VM** in the same subnet, mount the share, then run your existing Python download scripts.

```bash
az vm create -g $RG -n openrlhf-prep \
  --image Ubuntu2204 --size Standard_D8s_v5 \
  --vnet-name $VNET --subnet $SUBNET \
  --admin-username $ADMIN_USER --ssh-key-values $SSH_KEY_PATH \
  --public-ip-sku Standard

# --- SSH in ---
ssh -i ~/.ssh/azure_openrlhf ${ADMIN_USER}@<PREP_PUBLIC_IP>

sudo apt-get update && sudo apt-get install -y nfs-common git python3-pip

sudo mkdir -p $MOUNT_POINT
sudo mount ${NFS_IP}:${NFS_PATH} $MOUNT_POINT
sudo chown $USER:$USER $MOUNT_POINT

# Hugging Face & dataset downloads (reuse scripts)
pip install huggingface_hub transformers accelerate torch
export HF_HOME=$MOUNT_POINT/model-cache/huggingface
python3 <OpenRLHF-M>/examples/scripts/downloaders/download_mathv60k.py \
        --root_dir $MOUNT_POINT/datasets/VerMulti

exit
# Optional: delete prep VM
az vm delete -g $RG -n openrlhf-prep --yes --no-wait
```

---

## 5  Create Head & Worker GPU VMs

```bash
for VM in $HEAD_VM $WORKER_VM; do
  az vm create -g $RG -n $VM \
    --size $VM_SIZE \
    --image OpenLogic:0001-com-ubuntu-minimal-jammy:22_04-lts-gen2:latest \
    --admin-username $ADMIN_USER --ssh-key-values $SSH_KEY_PATH \
    --vnet-name $VNET --subnet $SUBNET \
    --accelerated-networking true \
    --public-ip-sku Standard \
    --custom-data cloud-init.txt      # optional, see below
done
```

`cloud-init.txt` (optional) can auto-install `nvidia-driver-535`, `docker`, and mount NFS via `/etc/fstab`.

---

### 5.1  Manual VM Bootstrap

SSH into each VM and run:

```bash
# ---------------- Common (head & worker) ----------------
sudo apt-get update
# NVIDIA driver & docker convenience script:
curl https://get.docker.com | sh
sudo systemctl enable --now docker

sudo apt-get install -y nfs-common
sudo mkdir -p $MOUNT_POINT
sudo mount ${NFS_IP}:${NFS_PATH} $MOUNT_POINT
echo "${NFS_IP}:${NFS_PATH} ${MOUNT_POINT} nfs defaults 0 0" \
 | sudo tee -a /etc/fstab

# Docker login + pull image
az acr login -n $ACR_NAME            # will use Managed Identity auth if enabled
sudo docker pull ${ACR_LOGIN_SERVER}/${IMAGE}
```

---

## 6  Run Containers & Start Ray

### Head Node

```bash
# On head VM
sudo docker run -d --name openrlhf-head \
  --gpus all --net host --shm-size 16g --ulimit memlock=-1 \
  -v ${MOUNT_POINT}:${MOUNT_POINT} \
  -e HF_HOME=${MOUNT_POINT}/model-cache/huggingface \
  -e WANDB_API_KEY=${WANDB_API_KEY} \
  ${ACR_LOGIN_SERVER}/${IMAGE} sleep infinity

# Inside container
sudo docker exec -it openrlhf-head bash
export ETH_IF=eth0                      # Azure intra-VNet interface
export HEAD_IP=$(hostname -I | awk '{print $1}')
ray stop || true
ray start --head --node-ip-address ${HEAD_IP} \
           --port=6379 --dashboard-port=8265
```

### Worker Node(s)

```bash
# On worker VM
sudo docker run -d --name openrlhf-worker \
  --gpus all --net host --shm-size 16g --ulimit memlock=-1 \
  -v ${MOUNT_POINT}:${MOUNT_POINT} \
  -e HF_HOME=${MOUNT_POINT}/model-cache/huggingface \
  -e WANDB_API_KEY=${WANDB_API_KEY} \
  ${ACR_LOGIN_SERVER}/${IMAGE} sleep infinity

# Inside container
sudo docker exec -it openrlhf-worker bash
export HEAD_IP=<HEAD_PRIVATE_IP>      # same subnet address
ray stop || true
ray start --address=${HEAD_IP}:6379
```

Set NCCL to prefer the VNet interface (optional but recommended):

```bash
export NCCL_SOCKET_IFNAME=eth0
export NCCL_DEBUG=INFO
```

---

## 7  Launch Training

SSH to the **head VM**, enter the container, then:

```bash
cd /app/OpenRLHF-M/examples/scripts/tests
bash ./train_grpo_ray_qwen2_5_vl_mathv60k_multinode_azure.sh
```

Adapt the script just like your RunPod/GCP versions:

```bash
# Example flags inside the script
--actor_num_nodes 1 \
--actor_num_gpus_per_node 4 \
--vllm_num_engines 4 \
--vllm_tensor_parallel_size 1 \
--runtime-env-json "{\"working_dir\": \"/app/OpenRLHF-M\"}" \
--save_steps 50 --max_ckpt_num 2 --disable_ds_ckpt --save_hf_ckpt
```

Check Ray dashboard at `http://<HEAD_PUBLIC_IP>:8265` (open port via NSG or use SSH tunnel).

---

## 8  Cleanup

```bash
az vm delete -g $RG -n $HEAD_VM --yes --no-wait
az vm delete -g $RG -n $WORKER_VM --yes --no-wait

# Delete storage (DANGER: data loss!)
az netappfiles volume delete -g $RG --account-name $ANF_ACC \
  --pool-name $ANF_POOL -n $ANF_VOL --yes   # or Azure Files deletion commands

# Finally
az group delete -n $RG --yes --no-wait
```

---

## Option 2 – Deploy with Azure CycleCloud

CycleCloud can template the entire cluster (scheduler + execute nodes) and automatically mount ANF.

1. Follow the official **"Plan Your CycleCloud Production Deployment"** checklist.  
2. In the cluster template (`.json`), add:

```json
"ondemandvm": "Standard_NC48ads_A100_v4",
"configure_nfs_client": {
  "server": "${NFS_IP}",
  "export_path": "${NFS_PATH}",
  "mount_point": "/mnt/nfs"
}
```
3. Upload the template, provision cluster, SSH to the scheduler, and submit your Ray head/worker scripts as **CycleCloud jobs**.

> CycleCloud handles restarts, scaling, and can re-create the cluster from template in a DR scenario.

---

## Option 3 – AKS + NetApp Files (CSI Driver)

If you prefer Kubernetes:

1. Create an **AKS** cluster with GPU node-pool (`--node-vm-size $VM_SIZE`) in the same VNet.
2. Install the **Azure NetApp Files CSI Driver** (`helm install anf-csi-driver ...`).
3. Define a `StorageClass` and `PersistentVolumeClaim` (`accessModes: ReadWriteMany`).
4. Mount the PVC in your Ray head / worker pod manifests exactly like the GCP GKE example – simply replace the Filestore CSI class with the ANF class name.
5. Submit Ray job (`ray job submit --address=...`) or Kubeflow/Python SDK as you prefer.

---

## Troubleshooting

* **NFS Mount Fails** – verify NSG rules allow **TCP 2049**, ensure VM subnet delegated to ANF (if required).
* **QuotaExceeded** – request GPU quota increase in the **Azure Portal > Usage + quotas**.
* **NCCL Hangs** – confirm `NCCL_SOCKET_IFNAME=eth0` (or correct interface), _not_ the Azure RDMA NIC name unless you enabled SR-IOV.
* **Ray "Too many open files"** – `ulimit -n 65536` before `ray start`.
* **Disk Space** – monitor ANF/Files volume with `df -h $MOUNT_POINT`; scale capacity online.

---

## Cost Tips

1. **Stop or deallocate** VMs when idle.  
2. Use **Auto-shrink** for Azure Files or resize ANF volumes down after training.  
3. Consider **Spot VMs** (`--priority Spot`) for non-critical experiments – 50-70 % cheaper but evictable.

---

## Conclusion

You can now run OpenRLHF-M on Azure with performance on par with GCP and RunPod setups. Replace components (e.g., ANF → Azure Files, VM sizes, CycleCloud vs manual) to fit your budget and operational style.

Happy training! 🚀