import gc
import torch
import spaces
import gradio as gr
from salient import extract_salient_object
from background import generate_background_image

@spaces.GPU
@torch.no_grad()
def salientx(img, salient_prompt, bg_prompt, semantic_type, reproducibility=False):
    # Extract salient object and mask
    salient_img, bg_mask = extract_salient_object(
        image_path=img,
        prompt=salient_prompt,
        semantic_type=semantic_type
    )

    # Cleanup to free CUDA memory
    del img  # Free input image
    gc.collect()
    torch.cuda.empty_cache()

    # Generate background
    bg_img = generate_background_image(
        salient_img=salient_img,
        bg_mask=bg_mask,
        prompt=bg_prompt,
        reproducibility=reproducibility
    )

    del salient_img, bg_mask
    gc.collect()
    torch.cuda.empty_cache()

    return bg_img

desc = """
<div><h2>SALIENTX</h2></div>
<div><h4>The image generation with salient features preserved.</h4></div>
<div><h4>It takes an input image from which only the salient feature is preserved. It uses two prompts: one for selecting the salient feature and one for background generation.</h4></div>
"""

with gr.Blocks(title="SalientX", css="""
    #submit-btn {
        height: 60px;
        font-size: 18px;
    }
""") as website:
    gr.Markdown(desc)
    with gr.Tab(label="SalientX"):
        with gr.Row():
            input_image = gr.Image(
                type='filepath',
                label='Input Image',
                image_mode='RGB',
                height=512,
                width=512
            )
            output_image = gr.Image(
                type='numpy',
                label='Output Image',
                height=512,
                width=512
            )
        with gr.Row():
            salient_prompt = gr.Textbox(
                label="Salient Prompt",
                info="Describe the object to segment in English."
            )
            background_prompt = gr.Textbox(
                label="Background Prompt",
                info="Describe the background scenario in English."
            )
        with gr.Row():
            semantic_type_img = gr.Checkbox(
                False,
                label="Semantic Level",
                info="Enable to segment body parts, background, or multiple objects."
            )
            reproducibility = gr.Checkbox(
                False,
                label="Reproducibility",
                info="Enable to make the background generation reproducible."
            )
        submit_image = gr.Button(
            value='Submit',
            variant='primary',
            size='lg',
            elem_id='submit-btn'
        )
        submit_image.click(
            fn=salientx,
            inputs=[input_image, salient_prompt, background_prompt, semantic_type_img, reproducibility],
            outputs=output_image
        )

        gr.Markdown("")
        gr.Examples(
            examples=[
                ["assets/deers.jpg", "select the left Deer", "place it in a garden", False, False],
                ["assets/lake.jpg", "select the lake", "lake in the forest", True, True],
            ],
            inputs=[input_image, salient_prompt, background_prompt, semantic_type_img, reproducibility],
            outputs=output_image
        )

website.launch(favicon_path="./assets/salientx.ico")