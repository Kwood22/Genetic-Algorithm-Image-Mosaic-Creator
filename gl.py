#! /usr/bin/env python
#  -*- coding: utf-8 -*-
#
import argparse
import cv2
import threading
import math
from decimal import Decimal
from copy import deepcopy
import os, os.path
from random import randint

from random import *
import numpy as np
from glob import glob
import sys

try:
    from Tkinter import *
except ImportError:
    from tkinter import *

try:
    import ttk
    py3 = False
except ImportError:
    import tkinter.ttk as ttk
    py3 = True


import proj2_support
from PIL import Image, ImageTk

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
#GA variables
pop_size = 10
tournament_size = 4 # 4-10
reproductionProb = 0.12#0.1
crossOverProb = 0.4#0.35
mutationProb= 0.01
mutationAmount = 5
#Global Variables

rows = args["rows"]
cols = args["cols"]
segImage=np.empty((rows,cols), dtype=object)

#directory paths
orig_scale_dir =os.path.join("project_lib","segmented_image") 
lib_dir=os.path.join("project_lib","img_lib")
output_dir=os.path.join("project_lib","output")


#program
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
            #cv2.imwrite(os.path.join(outputLib,'split'+str(imageCount)+'.jpg'),matrix[i][j])            
            imageCount +=1
        x+=w

    
    return matrix

def createFolder (folderName):
        dirName = os.path.join(output_dir,folderName)
        if not os.path.exists(dirName):
            os.makedirs(dirName)
        return dirName

setUpDirectories()
outputFolder = createFolder("Generations")
imageOrig= getOriginalImage(origImagePath)

#calculate the width and height of each segment of the image
segWidth = imageOrig.shape[0]/cols
segHeight = imageOrig.shape[1]/rows

#create library of resized images
imageLib = createImageLib(imageDir,lib_dir,segWidth,segHeight)

#segment original image for processing
segImage = segmentOriginalImage(imageOrig,orig_scale_dir)
genNumber =0

def vp_start_gui():
    '''Starting point when module is the main routine.'''
    global val, w, root
    root = Tk()
    top = New_Toplevel (root)
    proj2_support.init(root, top)
    root.mainloop()

w = None
def create_New_Toplevel(root, *args, **kwargs):
    '''Starting point when module is imported by another program.'''
    global w, w_win, rt
    rt = root
    w = Toplevel (root)
    top = New_Toplevel (w)
    proj2_support.init(w, top, *args, **kwargs)
    return (w, top)

def destroy_New_Toplevel():
    global w
    w.destroy()
    w = None


class New_Toplevel:
    def outputNewImage(self,imageOrig,segImage,outputLib):
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

    

    def initialPopulation(self,size,imgOrig):
        print("Creating Initial population...")
        
        i= 0 
        x=0
        imageCount = 0
        cnt =0
        pop = []
        for cnt in range(0, size):
            chromosome = self.createRandomImage(imgOrig)
            image = self.createImageFromMatrix(chromosome)
            #cv2.imwrite(os.path.join(outputFolder,'chromosome'+str(imageCount)+'.jpg'),image)
            pop.append(chromosome)
            imageCount += 1
        cv2.imwrite(os.path.join(outputFolder,'gen0.jpg'), self.createImageFromMatrix(pop[0]))
        return pop

    def createRandomImage(self,imgOrig):
        i= 0 
        x=0
        chromosome = np.empty((rows,cols), dtype=object)
        for i in range(0, rows):
            y =0
            for j in range(0,cols):
                shuffle(imageLib)
                chromosome[i,j]= imageLib[0]        
        return chromosome
        
    def calculateDifference(self,newVector, origVector):
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
        #dif = math.sqrt((2+(rD/256))*math.pow(R,2)+4*math.pow(G,2)+(2+((255-rD)/256))*math.pow(B,2))
        dif = math.sqrt (math.pow(R,2)+math.pow(G,2)+math.pow(B,2))
        return dif

    def fitness (self,imageOrig,population,segImage):
        fitnesses=[]
        i =0
        for chromosome in population:
            distances =0
            i=0
            for i in range(0,rows):
                j=0
                for j in range (0,cols):
                    img = self.createImageFromMatrix(chromosome)
                    avgColorNew = np.array(chromosome[i][j]).mean(axis=(0,1))
                    avgColorOrig = np.array(segImage[i][j]).mean(axis=(0,1))
                    distances += self.calculateDifference(avgColorNew,avgColorOrig)
            amount = rows*cols
            average_dist = distances/amount
            fitnesses.append (average_dist)
            
        return fitnesses

    def tournamentSelection(self,pop, popFitnesses,tournament_size):
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

    def crossOver (self,p1, p2):
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

    def mutation (self,p1):
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

    def createMatrixFormat (self,image):
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

    def createImageFromMatrix(self,mat):
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

    def createNewGeneration(self,genNumber,size,pop,popFitness,reproductionProb):
        
        #outputFolder = createFolder ("gen"+str(genNumber))
        #Copy all elements from old population
        newPop = deepcopy(pop)
        
        while len(newPop) < 2*len(pop):
            parent1 = newPop[self.tournamentSelection(pop,popFitness,tournament_size)]    
            operator = randint(1,3)
            operatorProb = randint(0,100)/100.0
            
            if (operator == 0): #crossover
                if (operatorProb <= crossOverProb):
                    parent2 =newPop[ self.tournamentSelection(pop,popFitness,tournament_size)]
                    offspring = self.crossOver (parent1,parent2)
                    newPop.append(offspring[0])
                    newPop.append(offspring[1])            
            elif (operator ==1 ): #mutation
                if (operatorProb <= mutationProb):
                    offspring = self.mutation(parent1)
                    newPop.append(offspring)            
            else: # randomly create new image
                if (operator <= reproductionProb):
                    newPop.append(self.createRandomImage(imageOrig))
        
        #calculate fitness of population, sort population by fitness
        fitnessPopulation = np.array(self.fitness(imageOrig,newPop,segImage))
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
        for chromosome in outputPop:
            img = self.createImageFromMatrix(chromosome)
            #cv2.imwrite(os.path.join(outputFolder,'chromosome'+str(imageCount)+'.jpg'),img)
            imageCount+=1
        cv2.imwrite(os.path.join(outputFolder,'gen'+str(genNumber)+'.jpg'),self.createImageFromMatrix( outputPop[0]))
        print("Gen "+str(genNumber)+" created")
        return outputPop

    def updateGUI(self,fitnessPopulation,population):
        self.fitnesslbl.configure(text='Fitness: '+str(fitnessPopulation[0]))
        #self.genNumber_11.configure(text='Generation: '+str(genNumber))
        b,g,r = cv2.split(self.createImageFromMatrix(population[0]))
        genIm = cv2.merge((r,g,b))
        genImg= ImageTk.PhotoImage(image=Image.fromarray(genIm)) 
        self.genImage.configure(image= genImg)

    def run(self):
        print("Done Setup....\n\n")
        #create an initial population
        #outputFolder = createFolder("Generations")
        population=self.initialPopulation (pop_size,imageOrig)
        fitnessPopulation = self.fitness(imageOrig,population,segImage)
        try:
            genNumber = 1
            while True:
                population=self.createNewGeneration(genNumber,pop_size,population,fitnessPopulation,0.1)
                fitnessPopulation = self.fitness(imageOrig,population,segImage)
                self.updateGUI(fitnessPopulation,population)
                genNumber+=1
        except KeyboardInterrupt:
            #cv2.waitKey
            cv2.destroyAllWindows()

    

    def __init__(self, top=None):
        '''This class configures and populates the toplevel window.
           top is the toplevel containing window.'''
        _bgcolor = '#d9d9d9'  # X11 color: 'gray85'
        _fgcolor = '#000000'  # X11 color: 'black'
        _compcolor = '#d9d9d9' # X11 color: 'gray85'
        _ana1color = '#d9d9d9' # X11 color: 'gray85' 
        _ana2color = '#d9d9d9' # X11 color: 'gray85' 
        font10 = "-family {DejaVu Sans} -size 13 -weight normal -slant"  \
            " roman -underline 0 -overstrike 0"
        font11 = "-family {DejaVu Sans} -size 12 -weight normal -slant"  \
            " roman -underline 0 -overstrike 0"
        font12 = "-family {DejaVu Sans} -size 14 -weight normal -slant"  \
            " roman -underline 0 -overstrike 0"
        font9 = "-family {DejaVu Sans} -size 17 -weight normal -slant "  \
            "roman -underline 0 -overstrike 0"

        top.geometry("1336x846+2366+83")
        top.title("COS 314 Project 2 (Kyle Wood 16087993)")
        top.configure(highlightcolor="black")
        
        b,g,r = cv2.split(imageOrig)
        img = cv2.merge((r,g,b))
        self.im = Image.fromarray(imageOrig)
        
        self.imgtk = ImageTk.PhotoImage(image=Image.fromarray(img)) 


        self.Labelframe1 = LabelFrame(top)
        self.Labelframe1.place(relx=0.02, rely=0.11, relheight=0.53
                , relwidth=0.47)
        self.Labelframe1.configure(relief=GROOVE)
        self.Labelframe1.configure(font=font12)
        self.Labelframe1.configure(text='''Original Image''')
        self.Labelframe1.configure(width=630)

        self.origImage = Label(self.Labelframe1)
        #self.origImage['image']=imgtk
        self.origImage.place(relx=0.03, rely=0.07, height=400, width=600, y=-18,)
        #self.origImage.configure(activebackground="#f9f9f9")
        self.origImage.configure(image= self.imgtk)

        self.menubar = Menu(top,font="TkMenuFont",bg=_bgcolor,fg=_fgcolor)
        top.configure(menu = self.menubar)



        self.Labelframe1_1 = LabelFrame(top)
        self.Labelframe1_1.place(relx=0.51, rely=0.11, relheight=0.53
                , relwidth=0.47)
        self.Labelframe1_1.configure(relief=GROOVE)
        self.Labelframe1_1.configure(font=font12)
        self.Labelframe1_1.configure(text='''Generated Mosaic''')
        self.Labelframe1_1.configure(width=630)

        self.genImage = Label(self.Labelframe1_1)
        self.genImage.place(relx=0.03, rely=0.09, height=400, width=600, y=-18)
        self.genImage.configure(activebackground="#f9f9f9")
        self.genImage.configure(text='''No image found''')

        self.fitnesslbl = Label(top)
        self.fitnesslbl.place(relx=0.88, rely=0.08, height=19, width=146)
        self.fitnesslbl.configure(activebackground="#f9f9f9")
        self.fitnesslbl.configure(font=font10)
        self.fitnesslbl.configure(text='''Fitness: #''')

        self.Labelframe1_7 = LabelFrame(top)
        self.Labelframe1_7.place(relx=0.02, rely=0.65, relheight=0.33
                , relwidth=0.79)
        self.Labelframe1_7.configure(relief=GROOVE)
        self.Labelframe1_7.configure(font=font12)
        self.Labelframe1_7.configure(text='''Genetic Algorithm Settings''')
        self.Labelframe1_7.configure(width=1060)

        self.Label4 = Label(self.Labelframe1_7)
        self.Label4.place(relx=0.03, rely=0.15, height=19, width=136, y=-18)
        self.Label4.configure(activebackground="#f9f9f9")
        self.Label4.configure(font=font11)
        self.Label4.configure(text='Population Size:')

        self.Label4_1 = Label(self.Labelframe1_7)
        self.Label4_1.place(relx=0.01, rely=0.35, height=19, width=156, y=-18)
        self.Label4_1.configure(activebackground="#f9f9f9")
        self.Label4_1.configure(font=font11)
        self.Label4_1.configure(text='''Selection Method:''')

        self.radTournament = Radiobutton(self.Labelframe1_7)
        self.radTournament.place(relx=0.16, rely=0.33, relheight=0.11
                , relwidth=0.2, y=-18, h=18)
        self.radTournament.configure(activebackground="#d9d9d9")
        self.radTournament.configure(font=font11)
        self.radTournament.configure(justify=LEFT)
        self.radTournament.configure(text='''Tournament Selection''')

        self.radOther = Radiobutton(self.Labelframe1_7)
        self.radOther.place(relx=0.13, rely=0.45, relheight=0.11, relwidth=0.21
                , y=-18, h=18)
        self.radOther.configure(activebackground="#d9d9d9")
        self.radOther.configure(font=font11)
        self.radOther.configure(justify=LEFT)
        self.radOther.configure(text='''Elitism Selection''')

        self.varCrossProb = IntVar(self.Labelframe1_7,value=round((crossOverProb)*100))
        self.crossoverProb = Scale(self.Labelframe1_7,variable=self.varCrossProb)
        self.crossoverProb.place(relx=0.56, rely=0.11, relwidth=0.42
                , relheight=0.0, height=65)
        self.crossoverProb.configure(activebackground="#d9d9d9")
        self.crossoverProb.configure(font=font11)
        self.crossoverProb.configure(label="Cross-over Probability (%)")
        self.crossoverProb.configure(length="442")
        self.crossoverProb.configure(orient="horizontal")
        self.crossoverProb.configure(troughcolor="#d9d9d9")
        self.crossoverProb.set(round((crossOverProb)*100))

        self.Label4_3 = Label(self.Labelframe1_7)
        self.Label4_3.place(relx=0.01, rely=0.78, height=19, width=146, y=-18)
        self.Label4_3.configure(activebackground="#f9f9f9")
        self.Label4_3.configure(font=font11)
        self.Label4_3.configure(text='''Tournament Size:''')

        self.varMutProb = IntVar(self.Labelframe1_7,value=round((mutationProb)*100))
        self.mutationProb = Scale(self.Labelframe1_7,variable=self.varMutProb)
        self.mutationProb.place(relx=0.56, rely=0.45, relwidth=0.42
                , relheight=0.0, height=65)
        self.mutationProb.configure(activebackground="#d9d9d9")
        self.mutationProb.configure(font=font11)
        self.mutationProb.configure(label="Mutation Probability (%)")
        self.mutationProb.configure(orient="horizontal")
        self.mutationProb.configure(troughcolor="#d9d9d9")

        self.Label4_4 = Label(self.Labelframe1_7)
        self.Label4_4.place(relx=0.55, rely=0.82, height=19, width=146, y=-18)
        self.Label4_4.configure(activebackground="#f9f9f9")
        self.Label4_4.configure(font=font11)
        self.Label4_4.configure(text='''Mutation Size:''')

        self.popSizeVar = StringVar(self.Labelframe1_7, value=str(pop_size))
        self.popSize = Entry(self.Labelframe1_7,textvariable=self.popSizeVar)
        self.popSize.place(relx=0.17, rely=0.15, relheight=0.09, relwidth=0.18
                , y=-18, h=18)
        self.popSize.configure(font=font11)
        self.popSize.configure(selectbackground=_bgcolor)
        self.popSize.configure(width=196)

        self.tourSizeVar = StringVar(self.Labelframe1_7, value=str(tournament_size))
        self.tourSize = Entry(self.Labelframe1_7, textvariable=self.tourSizeVar)
        self.tourSize.place(relx=0.17, rely=0.78, relheight=0.09, relwidth=0.18
                , y=-18, h=18)
        self.tourSize.configure(font=font11)
        self.tourSize.configure(selectbackground=_bgcolor)
        self.tourSize.configure(width=196)

        self.mutSizeVar = StringVar(self.Labelframe1_7, value=str(mutationAmount))
        self.mutationSize = Entry(self.Labelframe1_7,textvariable=self.mutSizeVar)
        self.mutationSize.place(relx=0.69, rely=0.82, relheight=0.09
                , relwidth=0.18, y=-18, h=18)
        self.mutationSize.configure(font=font11)
        self.mutationSize.configure(selectbackground=_bgcolor)
        self.mutationSize.configure(width=196)

        self.Labelframe1_8 = LabelFrame(top)
        self.Labelframe1_8.place(relx=0.82, rely=0.65, relheight=0.33
                , relwidth=0.16)
        self.Labelframe1_8.configure(relief=GROOVE)
        self.Labelframe1_8.configure(font=font12)
        self.Labelframe1_8.configure(text='''Options''')
        self.Labelframe1_8.configure(width=210)

        self.ButtonStart = Button(self.Labelframe1_8)
        self.ButtonStart.place(relx=0.19, rely=0.18, height=47, width=137, y=-18)

        self.ButtonStart.configure(activebackground="#8cd88f")
        self.ButtonStart.configure(background="#4b8c49")
        self.ButtonStart.configure(borderwidth="2")
        self.ButtonStart.configure(font=font10)
        self.ButtonStart.configure(takefocus="0")
        self.ButtonStart.configure(text='''Start''')
        self.ButtonStart.configure(command=self.run)

        self.ButtonStop = Button(self.Labelframe1_8)
        self.ButtonStop.place(relx=0.19, rely=0.45, height=47, width=137, y=-18)
        self.ButtonStop.configure(activebackground="#d87b75")
        self.ButtonStop.configure(background="#8c3f3f")
        self.ButtonStop.configure(borderwidth="2")
        self.ButtonStop.configure(font=font10)
        self.ButtonStop.configure(takefocus="0")
        self.ButtonStop.configure(text='''Stop''')

        self.ButtonOpen = Button(self.Labelframe1_8)
        self.ButtonOpen.place(relx=0.19, rely=0.75, height=47, width=137, y=-18)
        self.ButtonOpen.configure(activebackground="#d9d9d9")
        self.ButtonOpen.configure(background="#707a8c")
        self.ButtonOpen.configure(borderwidth="2")
        self.ButtonOpen.configure(font=font10)
        self.ButtonOpen.configure(takefocus="0")
        self.ButtonOpen.configure(text='''Open Folder''')

        self.Label3 = Label(top)
        self.Label3.place(relx=0.4, rely=0.02, height=60, width=242)
        self.Label3.configure(activebackground="#f9f9f9")
        self.Label3.configure(font=font9)
        self.Label3.configure(text='''COS 314 Project 2
Kyle Wood 16087993''')

        self.genNumber_11 = Label(top)
        self.genNumber_11.place(relx=0.86, rely=0.05, height=19, width=146)
        self.genNumber_11.configure(activebackground="#f9f9f9")
        self.genNumber_11.configure(font=font10)
        self.genNumber_11.configure(text='''Generation: #''')




if __name__ == '__main__':
    vp_start_gui()






