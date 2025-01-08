import imageio

def save_gif_imageio(rgb_array_list, output_path="output.gif", fps=30):
    with imageio.get_writer(output_path, mode="I", fps=fps) as writer:
        for frame in rgb_array_list:
            writer.append_data(frame)
    print(f"GIF saved at {output_path}")