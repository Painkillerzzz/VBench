from google import genai
from google.genai import types
from typing import List
import os
import tqdm
from captioning.prompts import CAPTION_PROMPT_VIDEO
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import json
import logging

from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GeminiCaptioner:

    def __init__(self, debug: bool = False):
        self.client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"), http_options={"timeout": 120000, "retry_options": {"attempts": 2}})
        self.model = "gemini-3.1-pro-preview"
        self.debug = debug

    def caption_image(self, video_path: str) -> str:
        response = self.client.models.generate_content(
            model = self.model,
            contents = types.Content(
                parts = [
                    types.Part(
                        text = CAPTION_PROMPT_VIDEO
                    ),
                    types.Part(
                        inline_data = types.Blob(data=open(video_path, "rb").read(), mime_type="video/mp4"),
                        video_metadata=types.VideoMetadata(fps=5),
                    ),
                ]
            )
        )

        if self.debug:
            logger.info(f"{video_path}: {response.text}")
        return response.text

    def parallel_caption(self, video_paths: List[str], num_workers: int = 10) -> List[str]:

        def _run_one(index: int, video_path: str):
            return index, self.caption_image(video_path)

        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = [executor.submit(_run_one, i, video_path) for i, video_path in enumerate(video_paths)]
            results = [None] * len(video_paths)
            for future in tqdm.tqdm(as_completed(futures), total=len(futures), desc="Captions"):
                try:
                    index, caption = future.result()
                    results[index] = caption
                except Exception as e:
                    print(f"Caption generation failed: {e}\nVideo path: {video_paths[index]}") 
                    continue
            return results


    def __call__(self, video_paths: List[str], num_workers: int = 10) -> None:

        captions = self.parallel_caption(video_paths, num_workers)

        for i, caption in enumerate(captions):

            original_video_path = video_paths[i]
            output_caption_path = original_video_path.parent / (original_video_path.stem + ".txt")
            with open(output_caption_path, "w", encoding="utf-8") as f:
                if caption is None:
                    continue
                f.write(caption)

        output_json_path = original_video_path.parent / "captions.json"

        results = []
        for i, caption in enumerate(captions):
            original_video_path = video_paths[i]
            output_caption_path = original_video_path.parent / (original_video_path.stem + ".txt")
            results.append({
                "video_path": str(original_video_path),
                "caption": caption
            })

        with open(output_json_path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=4)


if __name__ == "__main__":

    captioner = GeminiCaptioner(debug=True)
    video_paths = list(Path("dataset/v-0315").glob("*.mp4"))

    captioner(video_paths, num_workers=128)