import gradio as gr
from PIL import Image

from app.custom_models.mvimg_prediction import run_mvprediction
from app.utils import make_image_grid, split_image
from scripts.utils import save_glb_and_video

def concept_to_multiview(preview_img, input_processing, seed, guidance=1.):
    seed = int(seed)
    if preview_img is None:
        raise gr.Error("preview_img is none.")
    if isinstance(preview_img, str):
        preview_img = Image.open(preview_img)
    
    rgb_pils, front_pil = run_mvprediction(preview_img, remove_bg=input_processing, seed=seed, guidance_scale=guidance)
    rgb_pil = make_image_grid(rgb_pils, rows=2)
    return rgb_pil, front_pil

def concept_to_multiview_ui(concurrency_id="wkl"):
    with gr.Row():
        with gr.Column(scale=2):
            preview_img = gr.Image(type='pil', image_mode='RGBA', label='Frontview')
            input_processing = gr.Checkbox(
                value=True,
                label='Remove Background',
            )
            seed = gr.Slider(minimum=-1, maximum=1000000000, value=-1, step=1.0, label="seed")
            guidance = gr.Slider(minimum=1.0, maximum=5.0, value=1.0, label="Guidance Scale", step=0.5)
            run_btn = gr.Button('Generate Multiview', interactive=True)
        with gr.Column(scale=3):
            # export mesh display
            output_rgb = gr.Image(type='pil', label="RGB", show_label=True)
            output_front = gr.Image(type='pil', image_mode='RGBA', label="Frontview", show_label=True)
    run_btn.click(
        fn = concept_to_multiview,
        inputs=[preview_img, input_processing, seed, guidance],
        outputs=[output_rgb, output_front],
        concurrency_id=concurrency_id,
        api_name=False,
    )
    return output_rgb, output_front

from app.custom_models.normal_prediction import predict_normals
from scripts.multiview_inference import geo_reconstruct
def multiview_to_mesh_v2(rgb_pil, normal_pil, front_pil, do_refine=False, expansion_weight=0.1, init_type="std"):
    rgb_pils = split_image(rgb_pil, rows=2)
    if normal_pil is not None:
        normal_pil = split_image(normal_pil, rows=2)
    if front_pil is None:
        front_pil = rgb_pils[0]
    new_meshes = geo_reconstruct(rgb_pils, normal_pil, front_pil, do_refine=do_refine, predict_normal=normal_pil is None, expansion_weight=expansion_weight, init_type=init_type)
    ret_mesh, video = save_glb_and_video("/tmp/gradio/generated", new_meshes, with_timestamp=True, dist=3.5, fov_in_degrees=2 / 1.35, cam_type="ortho", export_video=False)
    return ret_mesh

def new_multiview_to_mesh_ui(concurrency_id="wkl"):
    with gr.Row():
        with gr.Column(scale=2):
            rgb_pil = gr.Image(type='pil', image_mode='RGB', label='RGB')
            front_pil = gr.Image(type='pil', image_mode='RGBA', label='Frontview(Optinal)')
            normal_pil = gr.Image(type='pil', image_mode='RGBA', label='Normal(Optinal)')
            do_refine = gr.Checkbox(
                value=False,
                label='Refine rgb',
                visible=False,
            )
            expansion_weight = gr.Slider(minimum=-1.0, maximum=1.0, value=0.1, step=0.1, label="Expansion Weight", visible=False)
            init_type = gr.Dropdown(choices=["std", "thin"], label="Mesh initialization", value="std", visible=False)
            run_btn = gr.Button('Generate 3D', interactive=True)
        with gr.Column(scale=3):
            # export mesh display
            output_mesh = gr.Model3D(value=None, label="mesh model", show_label=True)
    run_btn.click(
        fn = multiview_to_mesh_v2,
        inputs=[rgb_pil, normal_pil, front_pil, do_refine, expansion_weight, init_type],
        outputs=[output_mesh],
        concurrency_id=concurrency_id,
        api_name="multiview_to_mesh",
    )
    return rgb_pil, front_pil, output_mesh


#######################################
def create_step_ui(concurrency_id="wkl"):
    with gr.Tab(label="3D:concept_to_multiview"):
        concept_to_multiview_ui(concurrency_id)
    with gr.Tab(label="3D:new_multiview_to_mesh"):
        new_multiview_to_mesh_ui(concurrency_id)
