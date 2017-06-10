import os
import moviepy.editor as mpy


folder = os.path.join(os.path.curdir, 'movie', 'latent')
clip = mpy.ImageSequenceClip(folder, fps=1, load_images=True) #Accepts folders
#clip.write_gif("test.gif", fps =1)
clip.write_videofile('latent.mp4', fps=1, codec='mpeg4', audio=False)