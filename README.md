# Science Immersion Day

# Module 1: Machine Learning

Student Login:
URL: https://607117263726.signin.aws.amazon.com/console
Username: ecolab<nn>
Password: <coursePassword>

## Module 1.1: Inference
Services: `SageMaker`
Select: `Notebook Instances` from Sidebar

Click: `ecolab-nb<nn>`
Then: `Open`

Expect: Jupyter Notebook home page.
Click: `Files` tab
Click: `New` button.
Select: `conda_python3`

Paste: *imageClassify.py* CELL1 from Software section (below)
Click `>|` (run cell) button on toolbar
Expect: Lovely Swan Pic

Paste *imageClassify.py* CELL2 from Software section (below)
Click `>|` (run cell) button on toolbar
Expect: probability of swan

## Module 1.2: Training
Services: `Sagemaker` - or click on SageMaker browser tab.
Select: `Notebook Instances` from Sidebar

Click: `ecolab-nb00`
Then: `Open`

Expect: Jupyter Notebook home page.
Click: `Running` tab

Click: `sample-notebooks/introduction_to_amazon_algorithms/imageclassification_caltech/Image-classification-transfer-learning.ipynb`

Walkthrough: Training process, model creation and inference endpoint.


# Module 2: Statistical Analysis

Student Login:
URL: https://appstream2.ap-southeast-2.aws.amazon.com/userpools#/signin?ref=WHoMsegfU9
Username: craigar+ecolab<nn>@amazon.com
Password: T0ng@riro

Expect: AppStream Dashboard

## Module 2.1 RStudio

Click: Firefox
Click: Firefox +tab or New Tab
URL: http://10.0.0.254:8787

```
To paste into AppStream from physical device clipboard:
1. Click on Clipboard icon in AppStream toolbar
2. Select 'Paste to remote session'
3. Perform native paste key sequence for physical device (Command-V, Ctrl-V etc) into dialogue box
4. Select target field for paste in AppStream application
5. Click Ctrl-V (Windows paste)
```
Expect: RStudio Login appears
Username: hadoop
Password: hadoop

Expect: RStudio IDE
Click: New R Script on toolbar or under File menu


# Module 3: Cluster Computing

## Module 3.1: OpenFOAM

Student Login:
URL: https://appstream2.ap-southeast-2.aws.amazon.com/userpools#/signin?ref=WHoMsegfU9
Username: craigar+ecolab00@amazon.com
Password: T0ng@riro

Expect: AppStream Dashboard
Click: PuTTY

Click: MyFiles icon in AppStream toolbar
Select: Home Folder
Click: Upload Files
Select: EcoLabSYD.ppk

Host Name: 10.75.128.15
Auth: EcoLabSYD.ppk

Expect: Alces Welcome
Command: `alces gridware list`
Expect: List includes apps/openfoam/4.<x>

If not already installed:
Command: `alces gridware install apps/openfoam/4.1` assuming <x> is 1

Command: `alces session start gnome`
Read: <vncport> and <vncpassword>
Tip: Highlight vncpassword to save to clipboard

Click: Applications icon in AppStream toolbar
Select: VNC
URL: 10.75.128.15:<vncport>
Password: <vncpassword>
Tip: right-mouse-click then paste

Expect: Gnome Desktop
Select: Applications | Terminal
Command: `module load apps/openfoam`
Commands: If cavity tutorial not already prepared
```bash
cd $FOAM_TUTORIALS
ls
cp -r $FOAM_TUTORIALS/incompressible/icoFoam/cavity/cavity $HOME/.
```

Command: `cd ~\cavity`
Command: `blockMesh`
Expect: creating block mesh...patches...end
Command: `checkMesh`
Expect: Checking geometry... mesh ok
Command: `icoFoam`
Expect: solver information..end
Command: `paraFoam`
Expect: ParaView main window
Navigation:
* Mesh Parts - tick All - then Apply
* Choose U + Magnitude + Surface
* Time: advance to 5

Further experimentation:
http://docs.alces-flight.com/en/stable/getting-started/environment-usage/using-openfoam-with-alces-flight-compute.html




# Software

## imageClassify.py

### CELL 1
```python
import os
import urllib.request
# url = 'http://www.vision.caltech.edu/Image_Datasets/Caltech256/images/207.swan/207_0054.jpg'
# url = 'http://www.vision.caltech.edu/Image_Datasets/Caltech256/images/012.binoculars/012_0004.jpg'
# url = 'https://upload.wikimedia.org/wikipedia/commons/3/35/Mute_swan_Vrhnika.jpg'
url = 'https://upload.wikimedia.org/wikipedia/commons/2/23/Brown_teal_in_water.JPG'
file_name = url.split("/")[-1]
if not os.path.exists(file_name):
    urllib.request.urlretrieve(url, file_name)

from IPython.display import Image
Image(file_name)
```

### CELL 2
```python
import boto3
import json
import numpy as np

endpoint_name = "sagemaker-imageclassification-notebook-ep--2018-02-28-20-30-49"
runtime = boto3.Session().client(service_name='runtime.sagemaker')

with open(file_name, 'rb') as f:
    payload = f.read()
    payload = bytearray(payload)
response = runtime.invoke_endpoint(EndpointName=endpoint_name,
                                   ContentType='application/x-image',
                                   Body=payload)
result = response['Body'].read()
# result will be in json format and convert it to ndarray
result = json.loads(result)
# the result will output the probabilities for all classes
# find the class with maximum probability and print the class index
index = np.argmax(result)
object_categories = ['ak47', 'american-flag', 'backpack', 'baseball-bat', 'baseball-glove', 'basketball-hoop', 'bat', 'bathtub', 'bear', 'beer-mug', 'billiards', 'binoculars', 'birdbath', 'blimp', 'bonsai-101', 'boom-box', 'bowling-ball', 'bowling-pin', 'boxing-glove', 'brain-101', 'breadmaker', 'buddha-101', 'bulldozer', 'butterfly', 'cactus', 'cake', 'calculator', 'camel', 'cannon', 'canoe', 'car-tire', 'cartman', 'cd', 'centipede', 'cereal-box', 'chandelier-101', 'chess-board', 'chimp', 'chopsticks', 'cockroach', 'coffee-mug', 'coffin', 'coin', 'comet', 'computer-keyboard', 'computer-monitor', 'computer-mouse', 'conch', 'cormorant', 'covered-wagon', 'cowboy-hat', 'crab-101', 'desk-globe', 'diamond-ring', 'dice', 'dog', 'dolphin-101', 'doorknob', 'drinking-straw', 'duck', 'dumb-bell', 'eiffel-tower', 'electric-guitar-101', 'elephant-101', 'elk', 'ewer-101', 'eyeglasses', 'fern', 'fighter-jet', 'fire-extinguisher', 'fire-hydrant', 'fire-truck', 'fireworks', 'flashlight', 'floppy-disk', 'football-helmet', 'french-horn', 'fried-egg', 'frisbee', 'frog', 'frying-pan', 'galaxy', 'gas-pump', 'giraffe', 'goat', 'golden-gate-bridge', 'goldfish', 'golf-ball', 'goose', 'gorilla', 'grand-piano-101', 'grapes', 'grasshopper', 'guitar-pick', 'hamburger', 'hammock', 'harmonica', 'harp', 'harpsichord', 'hawksbill-101', 'head-phones', 'helicopter-101', 'hibiscus', 'homer-simpson', 'horse', 'horseshoe-crab', 'hot-air-balloon', 'hot-dog', 'hot-tub', 'hourglass', 'house-fly', 'human-skeleton', 'hummingbird', 'ibis-101', 'ice-cream-cone', 'iguana', 'ipod', 'iris', 'jesus-christ', 'joy-stick', 'kangaroo-101', 'kayak', 'ketch-101', 'killer-whale', 'knife', 'ladder', 'laptop-101', 'lathe', 'leopards-101', 'license-plate', 'lightbulb', 'light-house', 'lightning', 'llama-101', 'mailbox', 'mandolin', 'mars', 'mattress', 'megaphone', 'menorah-101', 'microscope', 'microwave', 'minaret', 'minotaur', 'motorbikes-101', 'mountain-bike', 'mushroom', 'mussels', 'necktie', 'octopus', 'ostrich', 'owl', 'palm-pilot', 'palm-tree', 'paperclip', 'paper-shredder', 'pci-card', 'penguin', 'people', 'pez-dispenser', 'photocopier', 'picnic-table', 'playing-card', 'porcupine', 'pram', 'praying-mantis', 'pyramid', 'raccoon', 'radio-telescope', 'rainbow', 'refrigerator', 'revolver-101', 'rifle', 'rotary-phone', 'roulette-wheel', 'saddle', 'saturn', 'school-bus', 'scorpion-101', 'screwdriver', 'segway', 'self-propelled-lawn-mower', 'sextant', 'sheet-music', 'skateboard', 'skunk', 'skyscraper', 'smokestack', 'snail', 'snake', 'sneaker', 'snowmobile', 'soccer-ball', 'socks', 'soda-can', 'spaghetti', 'speed-boat', 'spider', 'spoon', 'stained-glass', 'starfish-101', 'steering-wheel', 'stirrups', 'sunflower-101', 'superman', 'sushi', 'swan', 'swiss-army-knife', 'sword', 'syringe', 'tambourine', 'teapot', 'teddy-bear', 'teepee', 'telephone-box', 'tennis-ball', 'tennis-court', 'tennis-racket', 'theodolite', 'toaster', 'tomato', 'tombstone', 'top-hat', 'touring-bike', 'tower-pisa', 'traffic-light', 'treadmill', 'triceratops', 'tricycle', 'trilobite-101', 'tripod', 't-shirt', 'tuning-fork', 'tweezer', 'umbrella-101', 'unicorn', 'vcr', 'video-projector', 'washing-machine', 'watch-101', 'waterfall', 'watermelon', 'welding-mask', 'wheelbarrow', 'windmill', 'wine-bottle', 'xylophone', 'yarmulke', 'yo-yo', 'zebra', 'airplanes-101', 'car-side-101', 'faces-easy-101', 'greyhound', 'tennis-shoes', 'toad', 'clutter']
print("Result: label - " + object_categories[index] + ", probability - " + str(result[index]))

```
