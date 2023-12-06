import PIL
from PIL import Image, ImageEnhance
# pic11 = Image.open('./visible10/2014_03_06_cloud_0.png')
# converter = ImageEnhance.Color(pic11)
# pic1 = converter.enhance(0.5)

image_1 = Image.open('./visible2/2015_05_26.png')
# image_1 = Image.open('2014_03_06_cloud_0.png')
image_2 = Image.open('./visible2/2015_09_15.png')
# image_2 = Image.open('2016_07_01_cloud_0.png')
enhancer1 = ImageEnhance.Color(image_1)
pic1 = enhancer1.enhance(10)
# Enhance the colors
enhancer2 = ImageEnhance.Color(image_2)
pic2 = enhancer2.enhance(10)  # Replace 'factor' with a value >1 to enhance colors
# pic1 = enhanced_image[:,:,:3]
# Save the enhanced image
pic1.save("enhanced_visible_2_1.jpg")
pic2.save("enhanced_visible_2_2.jpg")