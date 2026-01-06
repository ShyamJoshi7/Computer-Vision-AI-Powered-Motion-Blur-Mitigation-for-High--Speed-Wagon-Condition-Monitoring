import glob

blur = glob.glob(r"E:\wagon\DeblurGANv2\dataset/blur/*")
sharp = glob.glob(r"E:\wagon\DeblurGANv2\dataset/sharp/*")

print("Blur images:", len(blur))
print("Sharp images:", len(sharp))
print("Sample blur:", blur[:3])
print("Sample sharp:", sharp[:3])
