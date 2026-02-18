from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
from diffusers import ShapEPipeline
from diffusers.utils import export_to_ply
import tempfile
import os
import numpy as np
import trimesh
import shutil
import re
from datetime import datetime

app = FastAPI(title="Text to 3D with Shap-E")


class GenerateRequest(BaseModel):
    title: str | None = None          # ✅ AÑADIDO
    prompt: str
    seed: int = 0
    guidance_scale: float = 15.0
    num_steps: int = 64


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
pipe = ShapEPipeline.from_pretrained("openai/shap-e").to(device)


def ply_to_glb(ply_path: str) -> str:
    mesh = trimesh.load(ply_path)

    rot_x = trimesh.transformations.rotation_matrix(-np.pi / 2, [1, 0, 0])
    mesh = mesh.apply_transform(rot_x)

    rot_y = trimesh.transformations.rotation_matrix(np.pi, [0, 1, 0])
    mesh = mesh.apply_transform(rot_y)

    with tempfile.NamedTemporaryFile(suffix=".glb", delete=False) as glb_file:
        mesh.export(glb_file.name, file_type="glb")
        return glb_file.name


def safe_slug(value: str) -> str:
    """
    Convierte a snake_case seguro: letras/números/_ y sin path traversal.
    """
    value = value.strip().lower()
    value = value.replace(" ", "_")
    value = re.sub(r"[^a-z0-9_]+", "", value)
    value = re.sub(r"_+", "_", value).strip("_")
    return value


@app.post("/generate")
def generate_3d(req: GenerateRequest):
    try:
        generator = torch.Generator(device=device).manual_seed(req.seed)

        result = pipe(
            req.prompt,
            generator=generator,
            guidance_scale=req.guidance_scale,
            num_inference_steps=req.num_steps,
            frame_size=256,
            output_type="mesh",
        )

        images = result.images
        if not images:
            raise RuntimeError("No se generó ninguna malla 3D")

        # Guardar temporal .ply
        with tempfile.NamedTemporaryFile(suffix=".ply", delete=False, mode="w+b") as ply_file:
            export_to_ply(images[0], ply_file.name)
            ply_path = ply_file.name

        # Convertir a .glb temporal
        glb_tmp_path = ply_to_glb(ply_path)

        # Borrar .ply temporal
        try:
            os.remove(ply_path)
        except:
            pass

        # Outputs
        output_dir = "/app/outputs"
        os.makedirs(output_dir, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # ✅ construir filename seguro
        title = safe_slug(req.title) if req.title else ""
        if title:
            filename = f"{title}_{timestamp}.glb"
        else:
            filename = f"model_{timestamp}.glb"

        saved_path = os.path.join(output_dir, filename)

        # Mover archivo temporal a outputs
        shutil.move(glb_tmp_path, saved_path)

        print(f"[Shap-E] Guardado: {saved_path}")

        return {
            "filename": filename,
            "path": saved_path,
            "prompt": req.prompt
        }

    except Exception as e:
        print(f"[Shap-E] ERROR: {e}")
        raise HTTPException(status_code=500, detail=str(e))
