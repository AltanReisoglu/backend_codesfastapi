from fastapi import APIRouter
from pydantic import BaseModel
from core.model import StoryGenerator

router = APIRouter(
    prefix="/story",
    tags=["story"]
)

class StoryRequest(BaseModel):
    theme: str = "fantasy"
datas={
    "fantasy": "In a land of magic and dragons,s a young hero embarks on a quest to find a legendary artifact that can save their village from an impending doom.",
    "sci-fi": "In a distant future, humanity has colonized the stars",
}
def get_story_by_theme(theme: str) -> str:
    return datas.get(theme, None)

@router.post("/{theme}")
def generate_story(req: StoryRequest,theme):
    req.theme = theme
    if get_story_by_theme(req.theme):
        return {"theme": req.theme, "story": get_story_by_theme(req.theme)}
    else:
        story = StoryGenerator.generate_story(theme=req.theme)
        datas[req.theme] = story
        return {"theme": req.theme, "story": story}

@router.get("/last")
def get_story():
    keys = list(datas.keys())
    values = list(datas.values())
    return {"theme": keys[-1], "story": values[-1]}

@router.get("/{theme}")
def get_story(theme: str):
    if get_story_by_theme(theme):
        return {"theme": theme, "story": get_story_by_theme(theme)}
    else:
        return {"error": "Story not found for the given theme."}
    

@router.delete("/{theme}",status_code=200)
def delete_story(theme: str):
    if get_story_by_theme(theme):
        del datas[theme]
        return {"message": f"Story with theme '{theme}' has been deleted."}
    else:
        return {"error": "Story not found for the given theme."}