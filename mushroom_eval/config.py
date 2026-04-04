"""Configuration for mushroom character video evaluation pipeline."""

from dataclasses import dataclass, field


@dataclass(frozen=True)
class GeminiConfig:
    """Gemini VLM evaluation configuration."""
    model: str = "gemini-3.1-pro-preview"
    max_concurrent: int = 100
    max_retries: int = 5
    retry_delay: float = 2.0
    timeout: float = 60.0


@dataclass(frozen=True)
class EvalConfig:
    """Overall evaluation configuration."""
    gemini: GeminiConfig = field(default_factory=GeminiConfig)
    video_dir: str = ""
    output_dir: str = "mushroom_eval_results"
    caption_file: str = ""  # JSON/CSV with video_id -> caption mapping
    batch_size: int = 50  # Number of videos to process before saving checkpoint


# VLM evaluation questions for mushroom character
# Each question: (id, text, is_positive)
# is_positive=True: "yes" = good; is_positive=False: "yes" = bad
VLM_QUESTIONS = [
    # V1: 角色活力与僵硬度
    ("Q1_limb_motion",
     "Does the mushroom spirit character (蘑菇TUTU) show clear limb or body movement (such as walking, jumping, waving, bouncing, or turning)?",
     True),
    ("Q2_natural_motion",
     "Does the mushroom spirit character (蘑菇TUTU) move in a lively, natural way, like a small animal? (as opposed to being stiff like a plush toy or doll)",
     True),
    ("Q3_mostly_still",
     "Does the mushroom spirit character (蘑菇TUTU) remain mostly still or frozen throughout most of the video?",
     False),
    # V2: 运动主体正确性
    ("Q4_correct_subject",
     "Is the mushroom spirit character (蘑菇TUTU) the main object that is moving in the video? (rather than the background or other objects moving while it stays still)",
     True),
    # V3: 物理合理性与异常物体
    ("Q5_strange_objects",
     "Do any strange, distorted, or unrecognizable objects or shapes appear in the video that should not be there?",
     False),
    ("Q6_physics",
     "Does the motion in the video follow basic physical laws? (no objects floating impossibly, no incorrect sticking/merging of objects)",
     True),
    # V4: 时间一致性
    ("Q7_size_consistency",
     "Does the mushroom spirit character (蘑菇TUTU) maintain a consistent body size throughout the video? (no sudden unreasonable size changes)",
     True),
    ("Q8_appearance_consistency",
     "Does the mushroom spirit character (蘑菇TUTU)'s appearance (color, texture, clothing) remain consistent throughout the video?",
     True),
    # V5: 运动连贯性
    ("Q9_motion_continuity",
     "Are the mushroom spirit character (蘑菇TUTU)'s movements and position changes smooth and continuous? (no sudden teleportation or jerky jumps)",
     True),
]

# Prompt template for Gemini
VLM_SYSTEM_PROMPT = """Evaluate this AI-generated video of 蘑菇TUTU (mushroom character). Answer ONLY "yes" or "no". Output EXACTLY in this format, nothing else:
Q1: yes
Q2: no
Q3: yes
...Q9: yes

Do NOT repeat questions. Do NOT add explanations. ONLY "QN: yes" or "QN: no"."""

VLM_QUESTION_PROMPT = "\n".join(
    f"Q{i+1}: {q[1]}" for i, q in enumerate(VLM_QUESTIONS)
)
