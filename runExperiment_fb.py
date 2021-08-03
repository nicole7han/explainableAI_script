from __future__ import absolute_import, division
from psychopy.data import TrialHandler, importConditions
from psychopy import locale_setup, sound, gui, visual, core, data, event, logging
from psychopy.constants import (NOT_STARTED, STARTED, PLAYING, PAUSED,
                                STOPPED, FINISHED, PRESSED, RELEASED, FOREVER)
import numpy as np  # whole numpy lib is available, prepend 'np.'
import pandas as pd
from numpy import (sin, cos, tan, log, log10, pi, average,
                    sqrt, std, deg2rad, rad2deg, linspace, asarray)
from numpy.random import random, randint, normal, shuffle
import os, sys, glob, json

# set up path
current_path = os.path.abspath(os.getcwd())
stim_path = "{}/Stimuli".format(current_path)
data_path = "{}/Data".format(current_path)
script_path = current_path


# set up stimuli block_path.xlsx
blocks = glob.glob("{}/block*".format(stim_path))
blocks = [f for f in blocks if "xlsx" not in f]
blocks.sort()
stim_info = pd.read_excel('{}/stim_info.xlsx'.format(stim_path))
for b in range(len(blocks)):
    df = pd.DataFrame(columns=['image', 'corrAns'])
    stims = glob.glob("{}/*.jpg".format(blocks[b]))

    for i in stims:
        df = df.append({'image':i, 'corrAns':1 if "target" in i else 0}, ignore_index=True)

    df.to_excel('{}/block{}_stim_path.xlsx'.format(stim_path, b+1), index=None)


W, H = 1920, 1080
v_W, v_H = 1000, 800
N = len(pd.read_excel('{}/block1_stim_path.xlsx'.format(stim_path))) #number of trials per block

# get subject info
myDlg = gui.Dlg(title="Target vs Distractor Learning")
myDlg.addField('Subject Number:')
myDlg.addField('Age:')
myDlg.addField('Gender:', choices=["Female", "Male", "Rather not to say"])
info = myDlg.show()  # show dialog and wait for OK or Cancel
if myDlg.OK:  # or if ok_data is not None
    expInfo = {'Participant': info[0], 'Age': info[1], 'Gender':info[2] }
else:
    print('user cancelled')


#check subject previous blocks
if os.path.isdir(data_path + os.sep + "Data"+ os.sep + 'subj' +expInfo['Participant'])==False:
    os.mkdir(data_path + os.sep + "Data"+ os.sep + 'subj' +expInfo['Participant'])
subj_files = glob.glob(data_path + os.sep + "Data" + os.sep + 'subj' + expInfo['Participant'] + os.sep +"*.xlsx")
num_blocks = len(subj_files)
continue_last = False
finished_trials = []
dataFile = pd.DataFrame(columns=["image", "corr_ans", "resp"]) #start a new dataframe
block_num = num_blocks

if num_blocks > 0:
    last_file = [f for f in subj_files if expInfo['Participant'] + '_{}'.format(num_blocks) in f]
    last_block = pd.read_excel(last_file[0])
    if len(last_block) == N: #complete
        block_num += 1
    else:
        continue_last = True
        finished_trials = last_block['image'].tolist()
        block_num = num_blocks
        dataFile = last_block  # read last block dataframe
else:
    block_num = 1
expInfo['Block_num'] = block_num

# Data file name stem = absolute path + name; later add .psyexp, .csv, .log, etc
filename = data_path + os.sep + u'data' + os.sep + 'subj' + expInfo['Participant'] + os.sep +'subj'+ '%s_%s_%s_%s' %(expInfo['Participant'], expInfo['Block_num'], expInfo['Age'], expInfo['Gender'])

endExpNow = False  # flag for 'escape' or other condition => quit the exp

# Setup the Window
win = visual.Window(size=(1200, 900),fullscr=False, allowGUI=True, winType='pyglet',
        monitor='testMonitor', units ='pix', screen=0)

""" Initialize Compoenents """
## instructions ##
instrText = visual.TextStim(win=win, name='instrText',
    text="Welcome to the Experiment! In this experiment you are trying to discriminate a target and a distractor stimuli.\
        \n  They will be presented in a random order. \
        \n\n \
        Don't feel frustrated if you get it wrong at the beginning because it is supposed to be challenging. \n \
        The feedback is supposed to help you learn as you do more trials.\
        \n\n \
        1) Press the space bar to start each trial. \
        \n\n \
        2) Click to respond.",
    font='Arial', 
    units='pix', pos=[0, 0], height=30, wrapWidth=850, ori=0, 
    color=[1,1,1], colorSpace='rgb', opacity=1,
    depth=0.0)

## examples ##
examText = visual.TextStim(win=win, name='examText',
    text="The target and the distractor vary in different features. \n"
         "Here are some examples.",
    font='Arial',
    units='pix', pos=[0, 450], height=30, wrapWidth=850, ori=0,
    color=[1,1,1], colorSpace='rgb', opacity=1,
    depth=0.0)

example = visual.ImageStim(
    win=win, image='{}/examples/example.jpg'.format(stim_path),
    name='target1', mask=None,
    pos=(-350, 0), size=[350, 350],
    colorSpace='rgb', opacity=1
)
length = visual.ImageStim(
    win=win, image='{}/examples/length.jpg'.format(stim_path),
    name='target1', mask=None,
    pos=(250, 260), size=[550, 260],
    colorSpace='rgb', opacity=1
)
width = visual.ImageStim(
    win=win, image='{}/examples/width.jpg'.format(stim_path),
    name='target1', mask=None,
    pos=(250, 0), size=[550, 260],
    colorSpace='rgb', opacity=1
)
angle = visual.ImageStim(
    win=win, image='{}/examples/angle.jpg'.format(stim_path),
    name='target1', mask=None,
    pos=(250, -260), size=[550, 260],
    colorSpace='rgb', opacity=1
)


## trial iteratior ##
trial_num = visual.TextStim(win=win, name='trial_num',
    text='Trial: 1/{}'.format(N),
    font='Arial',
    units='pix', pos=[-450, 400], height=40, wrapWidth=800, ori=0,
    color=[1, 1, 1], colorSpace='rgb', opacity=1,
    depth=0.0);


## response screen ##
target_button = visual.Rect(win=win, name='Target',
    pos=(-200, -60),width = 200, height = 80, opacity = .8,
    lineColor=(0, 142/255.0, 18/255.0), lineColorSpace='rgb',
    fillColor=(0, 142/255.0, 18/255.0), fillColorSpace='rgb')
target_text = visual.TextStim(win=win, name='presentTxt',
    text='Target',
    font='Arial',
    units='pix', pos=[-200, -60], height=28, wrapWidth=800, ori=0,
    color=[1,1,1], colorSpace='rgb', opacity=1,
    depth=0.0)

distractor_button = visual.Rect(win=win, name='Distractor',
    pos=(200, -60),width = 200, height = 80,opacity = .8,
    lineColor=(183/255.0, 28/255.0, 0), lineColorSpace='rgb',
    fillColor=(183/255.0, 28/255.0, 0), fillColorSpace='rgb')
distractor_text = visual.TextStim(win=win, name='absentTxt',
    text='Distractor',
    font='Arial',
    units='pix', pos=[200, -60], height=28, wrapWidth=800, ori=0,
    color=[1,1,1], colorSpace='rgb', opacity=1,
    depth=0.0)
    
slider = visual.Slider(win=win, name='confidence',
                       ticks=(1, 2, 3, 4, 5), granularity=1,
                       labels=['least confident', '      ', '      ', '      ', 'most confident'],
                       size=(500, 40), pos=[0, -140], units='pix',
                       style= ["whiteOnBlack"],color=[1,1,1],colorSpace='rgb',opacity=1,)

## Thanks screen ##
thanksText = visual.TextStim(win=win, name='thanksText',
    text='Thank you! You have completed this block. Press the space bar to exit.',
    font='arial',
    units='pix', pos=[0, 0], height=40, wrapWidth=800, ori=0, 
    color=[1, 1, 1], colorSpace='rgb', opacity=1,
    depth=0.0)


""" Present Instruction """
ContinueThisRoutine = True
keyResponse = event.BuilderKeyResponse()
trialClock = core.Clock()
t = 0
while ContinueThisRoutine :
    t = trialClock.getTime()
    instrText.setAutoDraw(True)
    win.flip()
    
    theseKeys = event.getKeys(keyList=['escape', 'space'])
    if "escape" in theseKeys:
        dlg = gui.Dlg(title='quit experiment?', screen=-1)
        dlg.addText('Are you sure you want to quit the experiment?')
        dlg.show()
        if dlg.OK:
            dataFile.to_excel(filename + '.xlsx', index = None)
            core.quit()
    if "space" in theseKeys:
        instrText.setAutoDraw(False)
        ContinueThisRoutine = False
        break

""" Present Examples """
ContinueThisRoutine = True
examText.setAutoDraw(True)
example.setAutoDraw(True)
length.setAutoDraw(True)
width.setAutoDraw(True)
angle.setAutoDraw(True)

while ContinueThisRoutine:
    theseKeys = event.getKeys(keyList=['escape', 'space'])
    if "escape" in theseKeys:
        dlg = gui.Dlg(title='quit experiment?', screen=-1)
        dlg.addText('Are you sure you want to quit the experiment?')
        dlg.show()
        if dlg.OK:
            dataFile.to_excel(filename + '.xlsx', index = None)
            core.quit()
    if "space" in theseKeys:
        examText.setAutoDraw(False)
        example.setAutoDraw(False)
        length.setAutoDraw(False)
        width.setAutoDraw(False)
        angle.setAutoDraw(False)
        ContinueThisRoutine = False
        break
    win.flip()


""" Set up and Present Trials in Random Order"""
stim_list = stim_path + os.sep + 'block{}'.format(expInfo['Block_num']) +'_stim_path.xlsx'
#set up handler to look after randomisation of conditions etc
trials = data.TrialHandler(nReps=1.0, method='random',
    extraInfo=expInfo, originPath=-1,
    trialList=data.importConditions(stim_list),
    seed=None, name='trials')

thisTrial = trials.trialList[0]  # so we can initialise stimuli with some values
stim_info = pd.read_excel('{}/stim_info.xlsx'.format(stim_path))
trial_iter = 1
if len(finished_trials) > 0:
    trial_iter = len(finished_trials) + 1

""" Present Trials """
for thisTrial in trials:
    ContinueThisRoutine = True
    PresentConfidence = False
    confidence = None
    if thisTrial != None and thisTrial['image'] not in finished_trials: #check if we have done this trial before
        keyResponse = event.BuilderKeyResponse()
        mouse = event.Mouse()
        ## present image ##
        trial_num.text = "Trial:{}/{}".format(trial_iter, N)
        imageName = thisTrial["image"]
        corrAns = thisTrial["corrAns"]
        img = visual.ImageStim(
            win=win, image=imageName,
            name='target1', mask=None,
            pos=(0, 250), size=[480, 480],
            colorSpace='rgb', opacity=1
        )
        trial_num.setAutoDraw(True)
        img.setAutoDraw(True)
        target_button.setAutoDraw(True)
        target_text.setAutoDraw(True)
        distractor_button.setAutoDraw(True)
        distractor_text.setAutoDraw(True)
        
        while ContinueThisRoutine:
            theseKeys = event.getKeys(keyList=['escape', 'space'])
            win.flip()
            if "escape" in theseKeys:
                dlg = gui.Dlg(title='quit experiment?', screen=-1)
                dlg.addText('Are you sure you want to quit the experiment?')
                dlg.show()
                if dlg.OK:
                    dataFile.to_excel(filename + '.xlsx', index = None)
                    core.quit()

            if "space" in theseKeys and PresentConfidence and confidence!=None:
                slider.reset()
                newrow = {'image': imageName, 'corr_ans': corrAns, 'resp': resp, 'conf': confidence}
                dataFile = dataFile.append(newrow, ignore_index=True)

                trial_num.setAutoDraw(False)
                img.setAutoDraw(False)
                target_button.setAutoDraw(False)
                target_text.setAutoDraw(False)
                distractor_button.setAutoDraw(False)
                distractor_text.setAutoDraw(False)
                # slider.setAutoDraw(False)
                # feedbk_text.setAutoDraw(False)
                win.flip()
                ContinueThisRoutine = False

            if mouse.isPressedIn(target_button): #select target or distractor
                resp = 1
                PresentConfidence = True
            elif mouse.isPressedIn(distractor_button):
                resp = 0
                PresentConfidence = True

            if PresentConfidence: #select confidence
                slider.draw()
                confidence = slider.getRating()

                if confidence != None:
                    if corrAns == resp == 1: # hit trials
                        feedback_text = "Correct! This is a target."
                        feedbk_text = visual.TextStim(win=win, name='feedback',
                                                      text=feedback_text,
                                                      font='Arial',
                                                      units='pix', pos=[0, -300], height=30, wrapWidth=900, ori=0,
                                                      color=[1, 1, 1], colorSpace='rgb', opacity=1,
                                                      depth=0.0)
                        feedbk_text.draw()

                    elif corrAns == 1 and resp == 0: # miss trials
                        feedback_text = "Incorrect! This is a target."
                        feedbk_text = visual.TextStim(win=win, name='feedback',
                                                      text=feedback_text,
                                                      font='Arial',
                                                      units='pix', pos=[0, -300], height=30, wrapWidth=900, ori=0,
                                                      color=[1, 1, 1], colorSpace='rgb', opacity=1,
                                                      depth=0.0)
                        feedbk_text.draw()

                    elif corrAns == 0 and resp == 1: # false positive trials
                        explan = stim_info.loc[stim_info['stim']==os.path.split(thisTrial['image'])[-1][:-4]]['feedback'].item()
                        feedback_text = "Incorrect! "+ explan.split('. ')[0] + '.\n' + \
                                        explan.split('. ')[1].split(',')[0] + ',\n' + \
                                        explan.split('. ')[1].split(',')[1] + ',\n' + \
                                        explan.split('. ')[1].split(',')[2]
                        feedbk_text = visual.TextStim(win=win, name='feedback',
                                                      text=feedback_text,
                                                      font='Arial',
                                                      units='pix', pos=[0, -300], height=30, wrapWidth=900, ori=0,
                                                      color=[1, 1, 1], colorSpace='rgb', opacity=1,
                                                      depth=0.0)
                        feedbk_text.draw()

                    elif corrAns == 0 and resp == 0: # correct rejection trials
                        explan = stim_info.loc[stim_info['stim']==os.path.split(thisTrial['image'])[-1][:-4]]['feedback'].item()
                        feedback_text = "Correct! " + explan.split('. ')[0] + '.\n' + \
                                        explan.split('. ')[1].split(',')[0] + ',\n' + \
                                        explan.split('. ')[1].split(',')[1] + ',\n' + \
                                        explan.split('. ')[1].split(',')[2]
                        feedbk_text = visual.TextStim(win=win, name='feedback',
                                                      text=feedback_text,
                                                      font='Arial',
                                                      units='pix', pos=[0, -300], height=30, wrapWidth=900, ori=0,
                                                      color=[1, 1, 1], colorSpace='rgb', opacity=1,
                                                      depth=0.0)
                        feedbk_text.draw()

        event.clearEvents()
        trial_iter = trial_iter+1
dataFile.to_excel(filename+'.xlsx', index = None)



""" Present thanksText """
ContinueThisRoutine = True
keyResponse = event.BuilderKeyResponse()
t = 0
while ContinueThisRoutine:
    thanksText.setAutoDraw(True)
    theseKeys = event.getKeys()
    if len(theseKeys) > 0:
        thanksText.setAutoDraw(False)
        ContinueThisRoutine = False
        break

# make sure everything is closed down
win.close()
core.quit()


