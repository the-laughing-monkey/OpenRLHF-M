#!/usr/bin/env bash

# Script to find US zones where a specific GCP machine type is available.
# Usage: ./find_zones_for_machine_type.sh <machine-type>
# Example: ./find_zones_for_machine_type.sh a2-highgpu-2g

set -eo pipefail

# Check if machine type argument is provided
if [ -z "$1" ]; then
    echo "Usage: $0 <machine-type>"
    echo "Example: $0 a2-highgpu-2g"
    exit 1
fi

MACHINE_TYPE="$1"

echo "Searching for US zones where machine type '${MACHINE_TYPE}' is available..."

# Use gcloud to list machine types, filtering by name and US zones, format to get only zone names
# Redirect stderr to /dev/null to suppress errors if the user doesn't have permission
# or if the machine type truly doesn't exist anywhere.
available_zones=$(gcloud compute machine-types list \
    --filter="name = '${MACHINE_TYPE}' AND zone ~ '^us-'" \
    --format='value(zone)' \
    2>/dev/null)

# Check if any zones were found
if [ -n "$available_zones" ]; then
    echo "--------------------------------------------------"
    echo "Machine type '${MACHINE_TYPE}' found in the following US zones:"
    echo "--------------------------------------------------"
    # Print each zone on a new line (gcloud output usually does this already)
    echo "$available_zones"
    echo "--------------------------------------------------"
else
    echo "--------------------------------------------------"
    echo "Machine type '${MACHINE_TYPE}' not found in any US zones."
    echo "Reasons could include:
    - The machine type name is incorrect.
    - The machine type is not offered in any US zone.
    - You lack permissions to list machine types in some/all zones."
    echo "--------------------------------------------------"
    exit 1 # Exit with error if not found
fi

exit 0 