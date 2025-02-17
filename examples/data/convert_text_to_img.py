import subprocess
from pathlib import Path
from tempfile import TemporaryDirectory
import PyPDF2
from pdf2image import convert_from_path
from PIL import ImageOps
import json
from tqdm import tqdm
SYSTEM_PROMPT = (
    "You are a helpful assistant good at solving math problems with step-by-step reasoning. You should "
    "first thinks about the reasoning process in the mind and then provides the user with the answer. Your answer must be in latex format and wrapped in $...$."
    "The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., "
    "<think> Since $1+1=2$, so the answer is $2$. </think><answer> $2$ </answer>, which means your output should start with <think> and end with </answer>."
)


class TypstToImageConverter:
    def __init__(
        self,
        output_dir="output",
        font_size="16pt",
        page_size=None,
        margins=None,
        dpi=300,
    ):
        self.output_dir = Path(output_dir)
        self.font_size = font_size
        self.page_size = page_size or {"width":"10cm","height":"10cm"}
        self.margins = margins or {"x": "0.5cm", "y":"0.5cm"}
        self.dpi = dpi

        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _build_typst_template(self, content):
        # 尺寸参数生成逻辑
        page_param = f"width:{self.page_size['width']},height:{self.page_size['height']}, margin:(x:{self.margins['x']},y:{self.margins['y']})"

        return f'#set page({page_param})\n#set text(size: {self.font_size})\n#let content = "{content}"\n#content'

    def process_question(self, content, question_id):
        with TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            
            typst_content = self._build_typst_template(content)
            typst_file = tmp_path / "temp.typ"
            with open(typst_file, "w", encoding="utf-8") as f:
                f.write(typst_content)
            pdf_file = tmp_path / "temp.pdf"
            try:
                subprocess.run(
                    [
                        "typst",
                        "compile",
                        str(typst_file),
                        str(pdf_file),
                    ],
                    check=True,
                    capture_output=True,
                    timeout=10,
                )
            except subprocess.CalledProcessError as e:
                print(f"Error compiling Typst for question {question_id}: {e.stderr.decode()}")
                return False
            except subprocess.TimeoutExpired:
                print(f"Typst compilation timeout for question {question_id}")
                return False

            # 检查PDF页数
            
            if not pdf_file.exists():
                return False

            try:
                with open(pdf_file, "rb") as f:
                    reader = PyPDF2.PdfReader(f)
                    if len(reader.pages) >= 2:
                        return False
            except PyPDF2.errors.PdfReadError:
                print(f"Invalid PDF generated for question {question_id}")
                return False

            # 转换为图像
            images = convert_from_path(pdf_file, dpi=self.dpi)
            for page_num, image in enumerate(images):
                output_file = self.output_dir / f"q{question_id}.jpg"
                bbox = get_bbox(pdf_file)
                width = image.width
                height = image.height
                bbox = (bbox[0]*width, bbox[1]*height, bbox[2]*width, bbox[3]*height)
                image = image.crop(bbox)
                image = ImageOps.expand(image, border=10, fill='white')
                image.save(output_file)


            return True

if __name__ == "__main__":
    with open("data/mathlv345.jsonl",'r') as f:
        data = []
        for d in f.readlines():
            data.append(json.loads(d))
    output_dir = "/root/projects/lmm-r1/data/mathlv345_img/"
    converter = TypstToImageConverter(
        output_dir=output_dir,
        font_size="16pt",
        page_size={"width":"15cm","height":"10cm"},
        dpi=110,
    )
    fail_questions = []
    final_data = []
    for idx, d in tqdm(enumerate(data)):
        answer = d['answer']
        if answer[0] != "$":
            answer = "$" + answer + "$"
        question = d['problem']
        question = question.replace("\\\\","\\")
        question = question.replace('"','\\"')


        success = converter.process_question(question, idx)
        if success:
            message = [
                {
                    "role":"system",
                    "content": SYSTEM_PROMPT
                },
                {
                    "role":"user",
                    "content": [
                        {
                            "type":"text",
                            "text":"Answer the question in the image."
                        },
                        {
                            "type":"image",
                            "image":"file://"+output_dir+f"q{idx}.jpg"
                        }
                    ]
                }
            ]
            final_data.append({"message":json.dumps(message,ensure_ascii=False),"answer":answer})
        else:
            fail_questions.append(question)
