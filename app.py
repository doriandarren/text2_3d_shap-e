from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel
import torch
from diffusers import ShapEPipeline
from diffusers.utils import export_to_ply
import tempfile
import os
import numpy as np
import trimesh
import shutil
from datetime import datetime

app = FastAPI(title="Text to 3D with Shap-E")


class GenerateRequest(BaseModel):
    prompt: str
    seed: int = 0
    guidance_scale: float = 15.0
    num_steps: int = 64


# ----------------------------
#  CARGA DEL MODELO AL ARRANCAR
# ----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

pipe = ShapEPipeline.from_pretrained("openai/shap-e")
pipe = pipe.to(device)


def ply_to_glb(ply_path: str) -> str:
    """Convierte el .ply a .glb con rotaciones útiles."""
    mesh = trimesh.load(ply_path)

    rot_x = trimesh.transformations.rotation_matrix(-np.pi / 2, [1, 0, 0])
    mesh = mesh.apply_transform(rot_x)

    rot_y = trimesh.transformations.rotation_matrix(np.pi, [0, 1, 0])
    mesh = mesh.apply_transform(rot_y)

    with tempfile.NamedTemporaryFile(suffix=".glb", delete=False) as glb_file:
        mesh.export(glb_file.name, file_type="glb")
        return glb_file.name


@app.post("/generate")
def generate_3d(req: GenerateRequest):
    """
    Recibe prompt → devuelve y guarda un archivo .glb.
    """
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

        # Asegurar carpeta outputs
        output_dir = "/app/outputs"
        os.makedirs(output_dir, exist_ok=True)

        timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
        filename = f"model_{timestamp}.glb"
        saved_path = os.path.join(output_dir, filename)

        # Mover archivo temporal a carpeta outputs (cross-FS safe)
        shutil.move(glb_tmp_path, saved_path)

        print(f"[Shap-E] Guardado: {saved_path}")

        return FileResponse(
            saved_path,
            media_type="model/gltf-binary",
            filename=filename,
        )

    except Exception as e:
        print(f"[Shap-E] ERROR: {e}")
        raise HTTPException(status_code=500, detail=str(e))
