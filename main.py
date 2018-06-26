import argparse
import cv2
import math
from decimal import Decimal
from copy import deepcopy
import os, os.path
from random import randint
from random import *
import numpy as np
from glob import glob
 
ap = argparse.ArgumentParser()
ap.add_argument("dir", help = "Path to the image libary")
ap.add_argument("img", help = "Path to image that want to match")
ap.add_argument("rows", help = "Path to image that want to match",type=int)
ap.add_argument("cols", help = "Path to image that want to match",type=int)
args = vars(ap.parse_args()) 

#debug info OpenCV version
print ("OpenCV version: " + cv2.__version__)
 
#image path and valid extensions
imageDir = args["dir"]
origImagePath = args["img"]
image_path_list = []
valid_image_extensions = [".jpg", ".jpeg", ".png", ".tif", ".tiff"] 
valid_image_extensions = [item.lower() for item in valid_image_extensions]


# Get original image
def getOriginalImage(pathOrig):
    imageOrig = cv2.imread(origImagePath)
    dim = (600,400)
    imageOrig = cv2.resize(imageOrig, dim)
    if imageOrig is not None:
        print("Loaded original image...")
    elif imageOrig is None:
        print ("Error loading: " + imageOrig)
    return imageOrig
        
def setUpDirectories():  
    print ("Creating Project library folders...")   
    proj_dir = "project_lib"
    if not os.path.exists(proj_dir):
        os.makedirs(proj_dir)
    lib_dir=os.path.join("project_lib","img_lib")
    orig_scale_dir =os.path.join("project_lib","segmented_image")     
    if not os.path.exists(lib_dir):
        os.makedirs(lib_dir)
    if not os.path.exists(orig_scale_dir):
        os.makedirs(orig_scale_dir)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    print ("Done creating folders")   


#create a list all files in directory 
def createImageLib (pathImageLib,outputLib,resizeWidth, resizeHeight): 
    lib = []   
    for file in os.listdir(imageDir):
        extension = os.path.splitext(file)[1]
        if extension.lower() not in valid_image_extensions:
            continue
        image_path_list.append(os.path.join(imageDir, file))
    i=0
    #loop through image_path_list to open each image
    for imagePath in image_path_list:
        image = cv2.imread(imagePath)
        if image is not None:
            dim = (resizeHeight,resizeWidth)
            resized = cv2.resize(image, dim)
            lib.append(resized)
            cv2.imwrite("{}/img{}.jpg".format(lib_dir,i), resized)
            i = i +1
        elif image is None:
            print ("Error loading: " + imagePath)
            #end this loop iteration and move on to next image
            continue
    return lib
    
#segment original image
def segmentOriginalImage(imageOrig,outputLib):
    matrix = np.empty((rows,cols), dtype=object)
    h = imageOrig.shape[0]/rows
    w = imageOrig.shape[1]/cols
    x=0
    imageCount = 0    
    for i in range(0, rows):
        y =0
        for j in range(0,cols):
            segment = imageOrig[y:y+h, x:x+w]
            y += h
            matrix[i][j]=segment
            cv2.imwrite(os.path.join(outputLib,'split'+str(imageCount)+'.jpg'),matrix[i][j])            
            imageCount +=1
        x+=w

    
    return matrix

def outputNewImage(imageOrig,segImage,outputLib):
    h = imageOrig.shape[0]/rows
    w = imageOrig.shape[1]/cols
    x=0
    imageCount = 0    
    segment = deepcopy(imageOrig)
    for i in range(0, rows):
        y =0
        for j in range(0,cols):
            segment[y:y+h, x:x+w]= segImage[i][j]
            y += h           
            imageCount +=1
        x+=w
    cv2.imwrite(os.path.join(outputLib,'final.jpg'),segment)
    print ("Done segmenting original")

def createFolder (folderName):
    dirName = os.path.join(output_dir,folderName)
    if not os.path.exists(dirName):
        os.makedirs(dirName)
    return dirName

def initialPopulation(size,imgOrig):
    print("Creating Initial population...")
    
    i= 0 
    x=0
    imageCount = 0
    cnt =0
    pop = []
    for cnt in range(0, size):
        chromosome = createRandomImage(imgOrig)
        image = createImageFromMatrix(chromosome)
        #cv2.imwrite(os.path.join(outputFolder,'chromosome'+str(imageCount)+'.jpg'),image)
        pop.append(chromosome)
        imageCount += 1
    cv2.imwrite(os.path.join(outputFolder,'gen0.jpg'), createImageFromMatrix(pop[0]))
    return pop

def createRandomImage(imgOrig):
    i= 0 
    x=0
    chromosome = np.empty((rows,cols), dtype=object)
    for i in range(0, rows):
        y =0
        for j in range(0,cols):
            shuffle(imageLib)
            chromosome[i,j]= imageLib[0]        
    return chromosome
    
def calculateDifference(newVector, origVector):
    rnew = newVector[0]
    bnew = newVector[1]
    gnew = newVector[2]

    rorig =origVector[0]
    borig =origVector[1]
    gorig =origVector[2]

    rD = (rnew+rorig)/2.0
    R = rorig- rnew
    G = gnew-gorig
    B = bnew-borig
    # Riemersma difference
    dif = math.sqrt((2+(rD/256))*math.pow(R,2)+4*math.pow(G,2)+(2+((255-rD)/256))*math.pow(B,2))
    #dif = math.sqrt (math.pow(R,2)+math.pow(G,2)+math.pow(B,2))
    return dif

def fitness (imageOrig,population,segImage):
    fitnesses=[]
    i =0
    for chromosome in population:
        distances =0
        i=0
        for i in range(0,rows):
            j=0
            for j in range (0,cols):
                img = createImageFromMatrix(chromosome)
                avgColorNew = np.array(chromosome[i][j]).mean(axis=(0,1))
                avgColorOrig = np.array(segImage[i][j]).mean(axis=(0,1))
                distances += calculateDifference(avgColorNew,avgColorOrig)
        amount = rows*cols
        average_dist = distances/amount
        fitnesses.append (average_dist)
        
    return fitnesses

def tournamentSelection(pop, popFitnesses,tournament_size):
    index  = randint(0,len(pop)-1)
    bestFitness = popFitnesses[index]
    bestIndx= index
    i=1
    for i in range (0,tournament_size):
        index  = randint(0,len(pop)-1)
        if (popFitnesses[index] < bestFitness):
            bestIndx = index
            bestFitness = popFitnesses[index]
    return bestIndx

def crossOver (p1, p2):
    #choose horizontal vs vertical crossover
    operator = randint(0, 1)
    offspring1 = deepcopy(p1)
    offspring2 = deepcopy(p2)
    if (operator == 0):  # vertical crossover
        crossOverPoint1 = randint(0, cols-1)
        crossOverPoint2 = randint(0, cols-1)
        while crossOverPoint2 == crossOverPoint1:
            crossOverPoint2 = randint(0, cols-1)

        #swap columns of p1 and p2 to create 2 offspring 
        offspring1 [crossOverPoint1,:] = deepcopy(p2[crossOverPoint1,:])
        offspring1 [crossOverPoint2,:] = deepcopy(p2[crossOverPoint2,:])
        offspring2 [crossOverPoint1,:] = deepcopy(p1[crossOverPoint1,:])
        offspring2 [crossOverPoint2,:] = deepcopy(p1[crossOverPoint2,:])
    else:  # horizontal crossover
        crossOverPoint1 = randint(0, rows-1)
        crossOverPoint2 = randint(0, rows-1)
        
        while crossOverPoint2 == crossOverPoint1:
            crossOverPoint2 = randint(0, rows-1)
        
        #swap rows of p1 and p2 to create 2 offspring 
        offspring1 [:,crossOverPoint1] = p2[:,crossOverPoint1]
        offspring1 [:,crossOverPoint2] = p2[:,crossOverPoint2]
        offspring2 [:,crossOverPoint1] = p1[:,crossOverPoint1]
        offspring2 [:,crossOverPoint2] = p1[:,crossOverPoint2]
    output = [offspring1,offspring2]
    return output

def mutation (p1):
    #choose horizontal vs vertical crossover
    mutationAmt = randint(0,mutationAmount)
    i = 0
    offspring = deepcopy(p1)
    # choose and place n random images 
    for i in range(0,mutationAmt):
        xIndx = randint(0,cols-1)
        yIndx = randint(0,rows-1)
        shuffle(imageLib)
        offspring[xIndx][yIndx]= imageLib[0]
    return offspring

def createMatrixFormat (image):
    mat=np.empty((rows,cols), dtype=object)
    h = image.shape[0]/rows
    w = image.shape[1]/cols 
    i= 0 
    x=0        
    for i in range(0, rows):
        y =0
        for j in range(0,cols):
            seg= image[y:y+h, x:x+w]
            mat[i][j]=seg
            y += h           
        x+=w
    return mat

def createImageFromMatrix(mat):
    h = imageOrig.shape[0]/rows
    w = imageOrig.shape[1]/cols 
    i= 0 
    x=0
    chromosome = deepcopy(imageOrig)
    for i in range(0, rows):
        y =0
        for j in range(0,cols):
            chromosome[y:y+h, x:x+w]= mat[i][j]
            y += h           
        x+=w
    return chromosome

def createNewGeneration(genNumber,size,pop,popFitness,reproductionProb):
    
    #outputFolder = createFolder ("gen"+str(genNumber))
    #Copy all elements from old population
    newPop = deepcopy(pop)
    
    while len(newPop) < 2*len(pop):
        parent1 = newPop[tournamentSelection(pop,popFitness,tournament_size)]    
        operator = randint(1,3)
        operatorProb = randint(0,100)/100.0
         
        if (operator == 0): #crossover
            if (operatorProb <= crossOverProb):
                parent2 =newPop[ tournamentSelection(pop,popFitness,tournament_size)]
                offspring = crossOver (parent1,parent2)
                newPop.append(offspring[0])
                newPop.append(offspring[1])            
        elif (operator ==1 ): #mutation
            if (operatorProb <= mutationProb):
                offspring = mutation(parent1)
                newPop.append(offspring)            
        else: # randomly create new image
            if (operator <= reproductionProb):
                #newPop.append(createRandomImage(imageOrig))
                newPop.append(population[0])
    
    #calculate fitness of population, sort population by fitness
    fitnessPopulation = np.array(fitness(imageOrig,newPop,segImage))
    sortedFitnessIndx = np.argsort(fitnessPopulation)
    newPop = np.array(newPop)
    newPop[sortedFitnessIndx]
    fitnessPopulation =fitnessPopulation[sortedFitnessIndx]
    newPop = newPop[sortedFitnessIndx]

    #New gen = top pop_size elements of population (reduce size back to pop_size)
    i=0 
    outputPop =[]
    for i in range(0,pop_size):
        outputPop.append(deepcopy(newPop[i]))
    imageCount =0
    print(fitnessPopulation)
    #sort(outputPop,fitnessPopulation)
    for chromosome in outputPop:
        img = createImageFromMatrix(chromosome)
        #cv2.imwrite(os.path.join(outputFolder,'chromosome'+str(imageCount)+'.jpg'),img)
        imageCount+=1
    cv2.imwrite(os.path.join(outputFolder,'gen'+str(genNumber)+'.jpg'),createImageFromMatrix( outputPop[0]))
    print("Gen "+str(genNumber)+" created")
    return outputPop

def sort(population,fitnessPopulation):
    pop = population
    fit = fitnessPopulation
    sortIndx = np.argsort(fit)
    fitnessPopulation = fit[sortIndx]
    population = pop[sortIndx]
    print(fitnessPopulation)
    return

#GA variables
pop_size = 20
tournament_size = 4 # 4-10
reproductionProb = 0.12#0.1
crossOverProb = 0.4#0.35
mutationProb= 0.01
mutationAmount = 10
#Global Variables

rows = args["rows"]
cols = args["cols"]
segImage=np.empty((rows,cols), dtype=object)

#directory paths
orig_scale_dir =os.path.join("project_lib","segmented_image") 
lib_dir=os.path.join("project_lib","img_lib")
output_dir=os.path.join("project_lib","output")

#program
setUpDirectories()
imageOrig= getOriginalImage(origImagePath)

#calculate the width and height of each segment of the image
segWidth = imageOrig.shape[0]/cols
segHeight = imageOrig.shape[1]/rows

#create library of resized images
imageLib = createImageLib(imageDir,lib_dir,segWidth,segHeight)

#segment original image for processing
segImage = segmentOriginalImage(imageOrig,orig_scale_dir)


outputNewImage(imageOrig,segImage,output_dir)
print("Done Setup....\n\n")
#cv2.imshow("b",segImage[1][1])
#create an initial population
outputFolder = createFolder("Generations")
population=initialPopulation (pop_size,imageOrig)

fitnessPopulation = fitness(imageOrig,population,segImage)
print(str(fitnessPopulation))
#tournamentSelection(population,fitnessPopulation,tournament_size)

#createNewGeneration(1,pop_size,population,fitnessPopulation,0.1)
try:
    genNumber = 1
    while True:
        population=createNewGeneration(genNumber,pop_size,population,fitnessPopulation,0.1)
        fitnessPopulation = fitness(imageOrig,population,segImage)
        genNumber+=1
except KeyboardInterrupt:
    #cv2.waitKey
    cv2.destroyAllWindows()
