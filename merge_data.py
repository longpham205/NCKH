import os
import shutil

root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__)))
path_merger = os.path.join(root_dir,'data/archive')
classes = os.listdir(path_merger)
path_target = os.path.join(root_dir,'data/targets')
path_input = os.path.join(root_dir,'data/inputs')

if not os.path.exists(path_input):
  os.mkdir(path_input)
  os.mkdir(path_target)

k = 1
print(classes)
for kind in classes:
  path = os.path.join(path_merger, str(kind))
  if os.path.isfile(path):
    continue
  for i, f in enumerate(os.listdir(path)):
    shutil.copyfile(os.path.join(path, f),
                  os.path.join(path_target, f))
    os.rename(os.path.join(path_target, f), os.path.join(path_target, f'IMG_{k}.jpg'))
    k += 1

print("Process N = ", k)