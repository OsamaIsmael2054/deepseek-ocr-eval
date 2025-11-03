import re
import os
import math
import torch
from PIL import Image, ImageOps
from typing import Optional
from transformers import TextStreamer

from .conversation import format_messages
from .transforms import BasicImageTransform
from .utils import dynamic_preprocess, text_encode, load_pil_images
from .utils import re_match, draw_bounding_boxes
from tqdm import tqdm

class NoEOSTextStreamer(TextStreamer):
    """Custom text streamer that replaces EOS token with newline."""
    
    def on_finalized_text(self, text: str, stream_end: bool = False):
        eos_text = self.tokenizer.decode([self.tokenizer.eos_token_id], skip_special_tokens=False)
        text = text.replace(eos_text, "\n")
        print(text, flush=True, end="")


def infer_with_image_object(
    model,
    tokenizer,
    image_object: Image.Image,
    prompt: str = '<image>\n<|grounding|>Convert the document to markdown. ',
    output_path: str = '',
    base_size: int = 1024,
    image_size: int = 640,
    crop_mode: bool = True,
    save_results: bool = False,
    eval_mode: bool = True
) -> Optional[str]:
    """
    Perform OCR inference on an image using DeepSeek OCR model.
    
    Args:
        model: The DeepSeek OCR model instance
        tokenizer: The tokenizer instance
        image_object: PIL Image object to process
        prompt: Prompt template for the model
        output_path: Path to save results
        base_size: Base size for image processing
        image_size: Target image size
        crop_mode: Whether to use dynamic cropping
        test_compress: Whether to test compression ratio
        save_results: Whether to save results to disk
        eval_mode: Whether to run in evaluation mode (no streaming)
    
    Returns:
        Model output text if eval_mode is True, None otherwise
    """
    # 1. Disable torch init
    model.disable_torch_init()

    # 2. Setup directories
    if output_path:
        os.makedirs(output_path, exist_ok=True)
        os.makedirs(f'{output_path}/images', exist_ok=True)

    # 3. Setup conversation
    if not prompt:
        raise ValueError('prompt is required!')
    
    images = []
    image_draw = None
    conversation = [
        {
            "role": "<|User|>",
            "content": f'{prompt}',
        },
        {"role": "<|Assistant|>", "content": ""},
    ]

    # 4. Format prompt
    prompt_text = format_messages(conversations=conversation, sft_format='plain', system_prompt='')
    patch_size = 16
    downsample_ratio = 4
    images = [image_object]

    # 5. Get image properties
    valid_img_tokens = 0
    ratio = 1
    if image_draw:
        w, h = image_draw.size
        ratio = 1 - ((max(w, h) - min(w, h)) / (max(w, h)))
    
    # 6. Initialize transforms and tokens
    image_transform = BasicImageTransform(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5), normalize=True)
    images_seq_mask = []

    image_token = '<image>'
    image_token_id = 128815
    text_splits = prompt_text.split(image_token)

    images_list, images_crop_list, images_seq_mask = [], [], []
    tokenized_str = []
    images_spatial_crop = []
    
    for text_sep, image in zip(text_splits, images):
        tokenized_sep = text_encode(tokenizer, text_sep, bos=False, eos=False)

        tokenized_str += tokenized_sep
        images_seq_mask += [False] * len(tokenized_sep)

        if crop_mode:
            if image.size[0] <= 640 and image.size[1] <= 640:
                crop_ratio = [1, 1]
            else:
                if crop_mode:
                    images_crop_raw, crop_ratio = dynamic_preprocess(image)
                else:
                    crop_ratio = [1, 1]
            
            # Process the global view
            global_view = ImageOps.pad(image, (base_size, base_size),
                                    color=tuple(int(x * 255) for x in image_transform.mean))
            
            if base_size == 1024:
                valid_img_tokens += int(256 * ratio)
            elif base_size == 1280:
                valid_img_tokens += int(400 * ratio)
            
            images_list.append(image_transform(global_view).to(torch.bfloat16))

            width_crop_num, height_crop_num = crop_ratio
            images_spatial_crop.append([width_crop_num, height_crop_num])
            
            if width_crop_num > 1 or height_crop_num > 1:
                # Process the local views
                for i in range(len(images_crop_raw)):
                    images_crop_list.append(image_transform(images_crop_raw[i]).to(torch.bfloat16))
            
            if image_size == 640:
                valid_img_tokens += len(images_crop_list) * 100

            num_queries = math.ceil((image_size // patch_size) / downsample_ratio)
            num_queries_base = math.ceil((base_size // patch_size) / downsample_ratio)
            tokenized_image = ([image_token_id] * num_queries_base + [image_token_id]) * num_queries_base
            tokenized_image += [image_token_id]
            if width_crop_num > 1 or height_crop_num > 1:
                tokenized_image += ([image_token_id] * (num_queries * width_crop_num) + [image_token_id]) * (
                            num_queries * height_crop_num)
            tokenized_str += tokenized_image
            images_seq_mask += [True] * len(tokenized_image)

        else:
            # Process the global view without cropping
            if image_size <= 640:
                image = image.resize((image_size, image_size))
            
            global_view = ImageOps.pad(image, (image_size, image_size),
                                    color=tuple(int(x * 255) for x in image_transform.mean))
            images_list.append(image_transform(global_view).to(torch.bfloat16))

            if base_size == 1024:
                valid_img_tokens += int(256 * ratio)
            elif base_size == 1280:
                valid_img_tokens += int(400 * ratio)
            elif base_size == 640:
                valid_img_tokens += int(100 * 1)
            elif base_size == 512:
                valid_img_tokens += int(64 * 1)

            width_crop_num, height_crop_num = 1, 1
            images_spatial_crop.append([width_crop_num, height_crop_num])

            # Add image tokens
            num_queries = math.ceil((image_size // patch_size) / downsample_ratio)
            tokenized_image = ([image_token_id] * num_queries + [image_token_id]) * num_queries
            tokenized_image += [image_token_id]
            tokenized_str += tokenized_image
            images_seq_mask += [True] * len(tokenized_image)

    tokenized_sep = text_encode(tokenizer, text_splits[-1], bos=False, eos=False)
    tokenized_str += tokenized_sep
    images_seq_mask += [False] * len(tokenized_sep)

    # Add the bos tokens
    bos_id = 0
    tokenized_str = [bos_id] + tokenized_str 
    images_seq_mask = [False] + images_seq_mask
    input_ids = torch.LongTensor(tokenized_str)
    images_seq_mask = torch.tensor(images_seq_mask, dtype=torch.bool)

    if len(images_list) == 0:
        images_ori = torch.zeros((1, 3, image_size, image_size))
        images_spatial_crop = torch.zeros((1, 2), dtype=torch.long)
        images_crop = torch.zeros((1, 3, base_size, base_size))
    else:
        images_ori = torch.stack(images_list, dim=0)
        images_spatial_crop = torch.tensor(images_spatial_crop, dtype=torch.long)
        if images_crop_list:
            images_crop = torch.stack(images_crop_list, dim=0)
        else:
            images_crop = torch.zeros((1, 3, base_size, base_size))

    # Generate output
    if not eval_mode:
        streamer = NoEOSTextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=False)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            with torch.no_grad():
                output_ids = model.generate(
                    input_ids.unsqueeze(0).cuda(),
                    images=[(images_crop.cuda(), images_ori.cuda())],
                    images_seq_mask=images_seq_mask.unsqueeze(0).cuda(),
                    images_spatial_crop=images_spatial_crop,
                    temperature=0.0,
                    eos_token_id=tokenizer.eos_token_id,
                    streamer=streamer,
                    max_new_tokens=8192,
                    no_repeat_ngram_size=20,
                    use_cache=True
                )
    else:
        with torch.autocast("cuda", dtype=torch.bfloat16):
            with torch.no_grad():
                output_ids = model.generate(
                    input_ids.unsqueeze(0).cuda(),
                    images=[(images_crop.cuda(), images_ori.cuda())],
                    images_seq_mask=images_seq_mask.unsqueeze(0).cuda(),
                    images_spatial_crop=images_spatial_crop,
                    temperature=0.0,
                    eos_token_id=tokenizer.eos_token_id,
                    max_new_tokens=8192,
                    no_repeat_ngram_size=35,
                    use_cache=True
                )
    if '<image>' in conversation[0]['content'] and save_results:
        outputs = tokenizer.decode(output_ids[0, input_ids.unsqueeze(0).cuda().shape[1]:])
        stop_str = '<｜end▁of▁sentence｜>'

        print('='*15 + 'save results:' + '='*15)
        
        if outputs.endswith(stop_str):
            outputs = outputs[:-len(stop_str)]
        outputs = outputs.strip()

        matches_ref, matches_images, mathes_other = re_match(outputs)
        # print(matches_ref)
        result = draw_bounding_boxes(image_draw, matches_ref, output_path)


        for idx, a_match_image in enumerate(tqdm(matches_images, desc="image")):
            outputs = outputs.replace(a_match_image, '![](images/' + str(idx) + '.jpg)\n')
        
        for idx, a_match_other in enumerate(tqdm(mathes_other, desc="other")):
            outputs = outputs.replace(a_match_other, '').replace('\\coloneqq', ':=').replace('\\eqqcolon', '=:')


        # if 'structural formula' in conversation[0]['content']:
        #     outputs = '<smiles>' + outputs + '</smiles>'
        with open(f'{output_path}/result.mmd', 'w', encoding = 'utf-8') as afile:
            afile.write(outputs)

        if 'line_type' in outputs:
            import matplotlib.pyplot as plt
            lines = eval(outputs)['Line']['line']

            line_type = eval(outputs)['Line']['line_type']
            # print(lines)

            endpoints = eval(outputs)['Line']['line_endpoint']

            fig, ax = plt.subplots(figsize=(3,3), dpi=200)
            ax.set_xlim(-15, 15)
            ax.set_ylim(-15, 15)

            for idx, line in enumerate(lines):
                try:
                    p0 = eval(line.split(' -- ')[0])
                    p1 = eval(line.split(' -- ')[-1])

                    if line_type[idx] == '--':
                        ax.plot([p0[0], p1[0]], [p0[1], p1[1]], linewidth=0.8, color='k')
                    else:
                        ax.plot([p0[0], p1[0]], [p0[1], p1[1]], linewidth = 0.8, color = 'k')

                    ax.scatter(p0[0], p0[1], s=5, color = 'k')
                    ax.scatter(p1[0], p1[1], s=5, color = 'k')
                except:
                    pass

            for endpoint in endpoints:

                label = endpoint.split(': ')[0]
                (x, y) = eval(endpoint.split(': ')[1])
                ax.annotate(label, (x, y), xytext=(1, 1), textcoords='offset points', 
                            fontsize=5, fontweight='light')
            

            plt.savefig(f'{output_path}/geo.jpg')
            plt.close()

        result.save(f"{output_path}/result_with_boxes.jpg")

    if '<image>' in conversation[0]['content'] and eval_mode:
        outputs = tokenizer.decode(output_ids[0, input_ids.unsqueeze(0).cuda().shape[1]:])
        stop_str = '<｜end▁of▁sentence｜>'
        if outputs.endswith(stop_str):
            outputs = outputs[:-len(stop_str)]
        outputs = outputs.strip()
        return outputs

    return None


def infer_with_image_path(model, tokenizer,  image_file:str, prompt='<image>\n<|grounding|>Convert the document to markdown. ', output_path = '', base_size=1024, image_size=640, crop_mode=True, save_results=False, eval_mode=False):
        model.disable_torch_init()

        os.makedirs(output_path, exist_ok=True)
        os.makedirs(f'{output_path}/images', exist_ok=True)

        conversation = [
            {
                "role": "<|User|>",
                "content": f'{prompt}',
                "images": [f'{image_file}'],
            },
            {"role": "<|Assistant|>", "content": ""},
        ]
        
        prompt = format_messages(conversations=conversation, sft_format='plain', system_prompt='')

        patch_size = 16
        downsample_ratio = 4
        images = load_pil_images(conversation)

        valid_img_tokens = 0
        ratio = 1

        image_draw = images[0].copy()

        w,h = image_draw.size
        # print(w, h)
        ratio = 1 - ((max(w, h) - min(w, h)) / (max(w, h)))
    

        image_transform=BasicImageTransform(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5), normalize=True)
        images_seq_mask = []

        image_token = '<image>'
        image_token_id = 128815
        text_splits = prompt.split(image_token)

        images_list, images_crop_list, images_seq_mask = [], [], []
        tokenized_str = []
        images_spatial_crop = []
        for text_sep, image in zip(text_splits, images):

            tokenized_sep = text_encode(tokenizer, text_sep, bos=False, eos=False)
            tokenized_str += tokenized_sep
            images_seq_mask += [False] * len(tokenized_sep)

            if crop_mode:

                if image.size[0] <= 640 and image.size[1] <= 640:
                    crop_ratio = [1, 1]

                else:
                    if crop_mode:
                        images_crop_raw, crop_ratio = dynamic_preprocess(image)
                    else:
                        crop_ratio = [1, 1]
                
                """process the global view"""
                # image = image.resize((base_size, base_size))
                global_view = ImageOps.pad(image, (base_size, base_size),
                                        color=tuple(int(x * 255) for x in image_transform.mean))
                
                if base_size == 1024:
                    valid_img_tokens += int(256 * ratio)
                elif base_size == 1280:
                    valid_img_tokens += int(400 * ratio)

                images_list.append(image_transform(global_view).to(torch.bfloat16))

                width_crop_num, height_crop_num = crop_ratio

                images_spatial_crop.append([width_crop_num, height_crop_num])
                
                
                if width_crop_num > 1 or height_crop_num > 1:
                    """process the local views"""
                    
                    for i in range(len(images_crop_raw)):
                        images_crop_list.append(image_transform(images_crop_raw[i]).to(torch.bfloat16))
                
                if image_size == 640:
                    valid_img_tokens += len(images_crop_list) * 100

                num_queries = math.ceil((image_size // patch_size) / downsample_ratio)
                num_queries_base = math.ceil((base_size // patch_size) / downsample_ratio)



                """add image tokens"""

                

                tokenized_image = ([image_token_id] * num_queries_base + [image_token_id]) * num_queries_base
                tokenized_image += [image_token_id]
                if width_crop_num > 1 or height_crop_num > 1:
                    tokenized_image += ([image_token_id] * (num_queries * width_crop_num) + [image_token_id]) * (
                                num_queries * height_crop_num)
                tokenized_str += tokenized_image
                images_seq_mask += [True] * len(tokenized_image)
                # num_image_tokens.append(len(tokenized_image))

            else:
                # best_width, best_height = self.image_size, self.image_size
                # print(image.size, (best_width, best_height)) # check the select_best_resolutions func

                """process the global view"""
                if image_size <= 640:
                    print('directly resize')
                    image = image.resize((image_size, image_size))
                # else:
                global_view = ImageOps.pad(image, (image_size, image_size),
                                        color=tuple(int(x * 255) for x in image_transform.mean))
                images_list.append(image_transform(global_view).to(torch.bfloat16))

                if base_size == 1024:
                    valid_img_tokens += int(256 * ratio)
                elif base_size == 1280:
                    valid_img_tokens += int(400 * ratio)
                elif base_size == 640:
                    valid_img_tokens += int(100 * 1)
                elif base_size == 512:
                    valid_img_tokens += int(64 * 1)

                width_crop_num, height_crop_num = 1, 1

                images_spatial_crop.append([width_crop_num, height_crop_num])


                """add image tokens"""
                num_queries = math.ceil((image_size // patch_size) / downsample_ratio)

                tokenized_image = ([image_token_id] * num_queries + [image_token_id]) * num_queries
                tokenized_image += [image_token_id]
                # tokenized_image += ([self.image_token_id] * (num_queries * width_crop_num) + [self.image_token_id]) * (
                #             num_queries * height_crop_num)
                tokenized_str += tokenized_image
                images_seq_mask += [True] * len(tokenized_image)
                # num_image_tokens.append(len(tokenized_image))
        

        """process the last text split"""
        tokenized_sep = text_encode(tokenizer, text_splits[-1], bos=False, eos=False)
        tokenized_str += tokenized_sep
        images_seq_mask += [False] * len(tokenized_sep)

        """add the bos tokens"""
        bos_id = 0
        tokenized_str = [bos_id] + tokenized_str 
        images_seq_mask = [False] + images_seq_mask



        input_ids = torch.LongTensor(tokenized_str)


        

        images_seq_mask = torch.tensor(images_seq_mask, dtype=torch.bool)


        if len(images_list) == 0:
            images_ori = torch.zeros((1, 3, image_size, image_size))
            images_spatial_crop = torch.zeros((1, 2), dtype=torch.long)
            images_crop = torch.zeros((1, 3, base_size, base_size))

        else:
            images_ori = torch.stack(images_list, dim=0)
            images_spatial_crop = torch.tensor(images_spatial_crop, dtype=torch.long)
            if images_crop_list:
                images_crop = torch.stack(images_crop_list, dim=0)
            else:
                images_crop = torch.zeros((1, 3, base_size, base_size))



        if not eval_mode:
            streamer = NoEOSTextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=False)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                with torch.no_grad():
                    output_ids = model.generate(
                        input_ids.unsqueeze(0).cuda(),
                        images=[(images_crop.cuda(), images_ori.cuda())],
                        images_seq_mask = images_seq_mask.unsqueeze(0).cuda(),
                        images_spatial_crop = images_spatial_crop,
                        temperature=0.0,
                        eos_token_id=tokenizer.eos_token_id,
                        streamer=streamer,
                        max_new_tokens=8192,
                        no_repeat_ngram_size = 20,
                        use_cache = True
                        )

        else:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                with torch.no_grad():
                    output_ids = model.generate(
                        input_ids.unsqueeze(0).cuda(),
                        images=[(images_crop.cuda(), images_ori.cuda())],
                        images_seq_mask = images_seq_mask.unsqueeze(0).cuda(),
                        images_spatial_crop = images_spatial_crop,
                        temperature=0.0,
                        eos_token_id=tokenizer.eos_token_id,
                        max_new_tokens=8192,
                        no_repeat_ngram_size = 35,
                        use_cache = True
                        )
                
        if '<image>' in conversation[0]['content'] and save_results:
            outputs = tokenizer.decode(output_ids[0, input_ids.unsqueeze(0).cuda().shape[1]:])
            stop_str = '<｜end▁of▁sentence｜>'

            print('='*15 + 'save results:' + '='*15)
            
            if outputs.endswith(stop_str):
                outputs = outputs[:-len(stop_str)]
            outputs = outputs.strip()

            matches_ref, matches_images, mathes_other = re_match(outputs)
            # print(matches_ref)
            result = draw_bounding_boxes(image_draw, matches_ref, output_path)


            for idx, a_match_image in enumerate(tqdm(matches_images, desc="image")):
                outputs = outputs.replace(a_match_image, '![](images/' + str(idx) + '.jpg)\n')
            
            for idx, a_match_other in enumerate(tqdm(mathes_other, desc="other")):
                outputs = outputs.replace(a_match_other, '').replace('\\coloneqq', ':=').replace('\\eqqcolon', '=:')


            # if 'structural formula' in conversation[0]['content']:
            #     outputs = '<smiles>' + outputs + '</smiles>'
            with open(f'{output_path}/result.mmd', 'w', encoding = 'utf-8') as afile:
                afile.write(outputs)

            if 'line_type' in outputs:
                import matplotlib.pyplot as plt
                lines = eval(outputs)['Line']['line']

                line_type = eval(outputs)['Line']['line_type']
                # print(lines)

                endpoints = eval(outputs)['Line']['line_endpoint']

                fig, ax = plt.subplots(figsize=(3,3), dpi=200)
                ax.set_xlim(-15, 15)
                ax.set_ylim(-15, 15)

                for idx, line in enumerate(lines):
                    try:
                        p0 = eval(line.split(' -- ')[0])
                        p1 = eval(line.split(' -- ')[-1])

                        if line_type[idx] == '--':
                            ax.plot([p0[0], p1[0]], [p0[1], p1[1]], linewidth=0.8, color='k')
                        else:
                            ax.plot([p0[0], p1[0]], [p0[1], p1[1]], linewidth = 0.8, color = 'k')

                        ax.scatter(p0[0], p0[1], s=5, color = 'k')
                        ax.scatter(p1[0], p1[1], s=5, color = 'k')
                    except:
                        pass

                for endpoint in endpoints:

                    label = endpoint.split(': ')[0]
                    (x, y) = eval(endpoint.split(': ')[1])
                    ax.annotate(label, (x, y), xytext=(1, 1), textcoords='offset points', 
                                fontsize=5, fontweight='light')
                

                plt.savefig(f'{output_path}/geo.jpg')
                plt.close()

            result.save(f"{output_path}/result_with_boxes.jpg")

        if '<image>' in conversation[0]['content'] and eval_mode:
            outputs = tokenizer.decode(output_ids[0, input_ids.unsqueeze(0).cuda().shape[1]:])
            stop_str = '<｜end▁of▁sentence｜>'
            if outputs.endswith(stop_str):
                outputs = outputs[:-len(stop_str)]
            # re_match
            outputs = outputs.strip()
            return outputs
        return None