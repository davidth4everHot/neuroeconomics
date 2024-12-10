#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This experiment was created using PsychoPy3 Experiment Builder (v2024.2.2),
    on December 09, 2024, at 19:18
If you publish work using this script the most relevant publication is:

    Peirce J, Gray JR, Simpson S, MacAskill M, Höchenberger R, Sogo H, Kastman E, Lindeløv JK. (2019) 
        PsychoPy2: Experiments in behavior made easy Behav Res 51: 195. 
        https://doi.org/10.3758/s13428-018-01193-y

"""

# --- Import packages ---
from psychopy import locale_setup
from psychopy import prefs
from psychopy import plugins
plugins.activatePlugins()
prefs.hardware['audioLib'] = 'ptb'
prefs.hardware['audioLatencyMode'] = '3'
from psychopy import sound, gui, visual, core, data, event, logging, clock, colors, layout, hardware
from psychopy.tools import environmenttools
from psychopy.constants import (NOT_STARTED, STARTED, PLAYING, PAUSED,
                                STOPPED, FINISHED, PRESSED, RELEASED, FOREVER, priority)

import numpy as np  # whole numpy lib is available, prepend 'np.'
from numpy import (sin, cos, tan, log, log10, pi, average,
                   sqrt, std, deg2rad, rad2deg, linspace, asarray)
from numpy.random import random, randint, normal, shuffle, choice as randchoice
import os  # handy system and path functions
import sys  # to get file system encoding

import psychopy.iohub as io
from psychopy.hardware import keyboard

# --- Setup global variables (available in all functions) ---
# create a device manager to handle hardware (keyboards, mice, mirophones, speakers, etc.)
deviceManager = hardware.DeviceManager()
# ensure that relative paths start from the same directory as this script
_thisDir = os.path.dirname(os.path.abspath(__file__))
# store info about the experiment session
psychopyVersion = '2024.2.2'
expName = 'onwuakpa_BOID_CUNGA.1.2025'  # from the Builder filename that created this script
# information about this experiment
expInfo = {
    'participant': f"{randint(0, 999999):06.0f}",
    'session': '001',
    'date|hid': data.getDateStr(),
    'expName|hid': expName,
    'psychopyVersion|hid': psychopyVersion,
}

# --- Define some variables which will change depending on pilot mode ---
'''
To run in pilot mode, either use the run/pilot toggle in Builder, Coder and Runner, 
or run the experiment with `--pilot` as an argument. To change what pilot 
#mode does, check out the 'Pilot mode' tab in preferences.
'''
# work out from system args whether we are running in pilot mode
PILOTING = core.setPilotModeFromArgs()
# start off with values from experiment settings
_fullScr = True
_winSize = [1440, 900]
# if in pilot mode, apply overrides according to preferences
if PILOTING:
    # force windowed mode
    if prefs.piloting['forceWindowed']:
        _fullScr = False
        # set window size
        _winSize = prefs.piloting['forcedWindowSize']

def showExpInfoDlg(expInfo):
    """
    Show participant info dialog.
    Parameters
    ==========
    expInfo : dict
        Information about this experiment.
    
    Returns
    ==========
    dict
        Information about this experiment.
    """
    # show participant info dialog
    dlg = gui.DlgFromDict(
        dictionary=expInfo, sortKeys=False, title=expName, alwaysOnTop=True
    )
    if dlg.OK == False:
        core.quit()  # user pressed cancel
    # return expInfo
    return expInfo


def setupData(expInfo, dataDir=None):
    """
    Make an ExperimentHandler to handle trials and saving.
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    dataDir : Path, str or None
        Folder to save the data to, leave as None to create a folder in the current directory.    
    Returns
    ==========
    psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    """
    # remove dialog-specific syntax from expInfo
    for key, val in expInfo.copy().items():
        newKey, _ = data.utils.parsePipeSyntax(key)
        expInfo[newKey] = expInfo.pop(key)
    
    # data file name stem = absolute path + name; later add .psyexp, .csv, .log, etc
    if dataDir is None:
        dataDir = _thisDir
    filename = u'data/%s_%s_%s' % (expInfo['participant'], expName, expInfo['date'])
    # make sure filename is relative to dataDir
    if os.path.isabs(filename):
        dataDir = os.path.commonprefix([dataDir, filename])
        filename = os.path.relpath(filename, dataDir)
    
    # an ExperimentHandler isn't essential but helps with data saving
    thisExp = data.ExperimentHandler(
        name=expName, version='',
        extraInfo=expInfo, runtimeInfo=None,
        originPath='C:\\Users\\david\\OneDrive\\Documents\\2024\\psycho.py\\onwuakpa_BOID_CUNGA.1.2025_lastrun.py',
        savePickle=True, saveWideText=True,
        dataFileName=dataDir + os.sep + filename, sortColumns='time'
    )
    thisExp.setPriority('thisRow.t', priority.CRITICAL)
    thisExp.setPriority('expName', priority.LOW)
    # return experiment handler
    return thisExp


def setupLogging(filename):
    """
    Setup a log file and tell it what level to log at.
    
    Parameters
    ==========
    filename : str or pathlib.Path
        Filename to save log file and data files as, doesn't need an extension.
    
    Returns
    ==========
    psychopy.logging.LogFile
        Text stream to receive inputs from the logging system.
    """
    # set how much information should be printed to the console / app
    if PILOTING:
        logging.console.setLevel(
            prefs.piloting['pilotConsoleLoggingLevel']
        )
    else:
        logging.console.setLevel('warning')
    # save a log file for detail verbose info
    logFile = logging.LogFile(filename+'.log')
    if PILOTING:
        logFile.setLevel(
            prefs.piloting['pilotLoggingLevel']
        )
    else:
        logFile.setLevel(
            logging.getLevel('info')
        )
    
    return logFile


def setupWindow(expInfo=None, win=None):
    """
    Setup the Window
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    win : psychopy.visual.Window
        Window to setup - leave as None to create a new window.
    
    Returns
    ==========
    psychopy.visual.Window
        Window in which to run this experiment.
    """
    if PILOTING:
        logging.debug('Fullscreen settings ignored as running in pilot mode.')
    
    if win is None:
        # if not given a window to setup, make one
        win = visual.Window(
            size=_winSize, fullscr=_fullScr, screen=0,
            winType='pyglet', allowStencil=False,
            monitor='testMonitor', color=[0,0,0], colorSpace='rgb',
            backgroundImage='', backgroundFit='none',
            blendMode='avg', useFBO=True,
            units='height', 
            checkTiming=False  # we're going to do this ourselves in a moment
        )
    else:
        # if we have a window, just set the attributes which are safe to set
        win.color = [0,0,0]
        win.colorSpace = 'rgb'
        win.backgroundImage = ''
        win.backgroundFit = 'none'
        win.units = 'height'
    win.mouseVisible = False
    win.hideMessage()
    # show a visual indicator if we're in piloting mode
    if PILOTING and prefs.piloting['showPilotingIndicator']:
        win.showPilotingIndicator()
    
    return win


def setupDevices(expInfo, thisExp, win):
    """
    Setup whatever devices are available (mouse, keyboard, speaker, eyetracker, etc.) and add them to 
    the device manager (deviceManager)
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    win : psychopy.visual.Window
        Window in which to run this experiment.
    Returns
    ==========
    bool
        True if completed successfully.
    """
    # --- Setup input devices ---
    ioConfig = {}
    
    # Setup iohub keyboard
    ioConfig['Keyboard'] = dict(use_keymap='psychopy')
    
    # Setup iohub experiment
    ioConfig['Experiment'] = dict(filename=thisExp.dataFileName)
    
    # Start ioHub server
    ioServer = io.launchHubServer(window=win, **ioConfig)
    
    # store ioServer object in the device manager
    deviceManager.ioServer = ioServer
    
    # create a default keyboard (e.g. to check for escape)
    if deviceManager.getDevice('defaultKeyboard') is None:
        deviceManager.addDevice(
            deviceClass='keyboard', deviceName='defaultKeyboard', backend='iohub'
        )
    if deviceManager.getDevice('Forward2baseline_instruct') is None:
        # initialise Forward2baseline_instruct
        Forward2baseline_instruct = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='Forward2baseline_instruct',
        )
    if deviceManager.getDevice('Forward2baseline') is None:
        # initialise Forward2baseline
        Forward2baseline = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='Forward2baseline',
        )
    if deviceManager.getDevice('Forward2task1') is None:
        # initialise Forward2task1
        Forward2task1 = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='Forward2task1',
        )
    if deviceManager.getDevice('Forward2Xtrainfo_ID') is None:
        # initialise Forward2Xtrainfo_ID
        Forward2Xtrainfo_ID = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='Forward2Xtrainfo_ID',
        )
    if deviceManager.getDevice('Forward2donate') is None:
        # initialise Forward2donate
        Forward2donate = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='Forward2donate',
        )
    if deviceManager.getDevice('Forward2contribution') is None:
        # initialise Forward2contribution
        Forward2contribution = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='Forward2contribution',
        )
    if deviceManager.getDevice('key_resp') is None:
        # initialise key_resp
        key_resp = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='key_resp',
        )
    if deviceManager.getDevice('key_resp_2') is None:
        # initialise key_resp_2
        key_resp_2 = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='key_resp_2',
        )
    # return True if completed successfully
    return True

def pauseExperiment(thisExp, win=None, timers=[], playbackComponents=[]):
    """
    Pause this experiment, preventing the flow from advancing to the next routine until resumed.
    
    Parameters
    ==========
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    win : psychopy.visual.Window
        Window for this experiment.
    timers : list, tuple
        List of timers to reset once pausing is finished.
    playbackComponents : list, tuple
        List of any components with a `pause` method which need to be paused.
    """
    # if we are not paused, do nothing
    if thisExp.status != PAUSED:
        return
    
    # start a timer to figure out how long we're paused for
    pauseTimer = core.Clock()
    # pause any playback components
    for comp in playbackComponents:
        comp.pause()
    # make sure we have a keyboard
    defaultKeyboard = deviceManager.getDevice('defaultKeyboard')
    if defaultKeyboard is None:
        defaultKeyboard = deviceManager.addKeyboard(
            deviceClass='keyboard',
            deviceName='defaultKeyboard',
            backend='ioHub',
        )
    # run a while loop while we wait to unpause
    while thisExp.status == PAUSED:
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=['escape']):
            endExperiment(thisExp, win=win)
        # sleep 1ms so other threads can execute
        clock.time.sleep(0.001)
    # if stop was requested while paused, quit
    if thisExp.status == FINISHED:
        endExperiment(thisExp, win=win)
    # resume any playback components
    for comp in playbackComponents:
        comp.play()
    # reset any timers
    for timer in timers:
        timer.addTime(-pauseTimer.getTime())


def run(expInfo, thisExp, win, globalClock=None, thisSession=None):
    """
    Run the experiment flow.
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    psychopy.visual.Window
        Window in which to run this experiment.
    globalClock : psychopy.core.clock.Clock or None
        Clock to get global time from - supply None to make a new one.
    thisSession : psychopy.session.Session or None
        Handle of the Session object this experiment is being run from, if any.
    """
    # mark experiment as started
    thisExp.status = STARTED
    # make sure variables created by exec are available globally
    exec = environmenttools.setExecEnvironment(globals())
    # get device handles from dict of input devices
    ioServer = deviceManager.ioServer
    # get/create a default keyboard (e.g. to check for escape)
    defaultKeyboard = deviceManager.getDevice('defaultKeyboard')
    if defaultKeyboard is None:
        deviceManager.addDevice(
            deviceClass='keyboard', deviceName='defaultKeyboard', backend='ioHub'
        )
    eyetracker = deviceManager.getDevice('eyetracker')
    # make sure we're running in the directory for this experiment
    os.chdir(_thisDir)
    # get filename from ExperimentHandler for convenience
    filename = thisExp.dataFileName
    frameTolerance = 0.001  # how close to onset before 'same' frame
    endExpNow = False  # flag for 'escape' or other condition => quit the exp
    # get frame duration from frame rate in expInfo
    if 'frameRate' in expInfo and expInfo['frameRate'] is not None:
        frameDur = 1.0 / round(expInfo['frameRate'])
    else:
        frameDur = 1.0 / 60.0  # could not measure, so guess
    
    # Start Code - component code to be run after the window creation
    
    # --- Initialize components for Routine "Welcome_scr" ---
    Welcome_screen = visual.TextStim(win=win, name='Welcome_screen',
        text='Welcome, and congratulations on earning the right to participate in this study.\nYou will be asked by the computer to do a couple of tasks. Please pay attention, and if you have any questions, raise your hand, and we will attend to you. \n\nPlease press the space bar to continue.',
        font='Arial',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    Forward2baseline_instruct = keyboard.Keyboard(deviceName='Forward2baseline_instruct')
    
    # --- Initialize components for Routine "Instruction4baseline" ---
    Guide4baseline = visual.TextStim(win=win, name='Guide4baseline',
        text='Now, we will take the baseline. Please focus on the white object at the center of the screen, and do the best you can to minimize any movements. It will last for 5 minutes. \n\nPress the SPACE bar to continue.',
        font='Arial',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    Forward2baseline = keyboard.Keyboard(deviceName='Forward2baseline')
    
    # --- Initialize components for Routine "Baseline" ---
    Baseline_fixation = visual.ImageStim(
        win=win,
        name='Baseline_fixation', 
        image='Whiteobject.jpg', mask=None, anchor='center',
        ori=0.0, pos=(0, 0), draggable=False, size=(0.5, 0.5),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=0.0)
    
    # --- Initialize components for Routine "Instruct4tasks" ---
    Guide2tasks_instructions = visual.TextStim(win=win, name='Guide2tasks_instructions',
        text='Now, we will ask you to do a couple of tasks and answer some questions. Please do them to the best of your ability.\n\nPlease press the SPACE bar to continue.',
        font='Arial',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    Forward2task1 = keyboard.Keyboard(deviceName='Forward2task1')
    
    # --- Initialize components for Routine "Task1" ---
    Intro_info_ID = visual.TextStim(win=win, name='Intro_info_ID',
        text='Please open the envelope, and identify the name on the envelope. Please say the name loud to yourself.\n\nPress the SPACE bar to continue.',
        font='Arial',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    Forward2Xtrainfo_ID = keyboard.Keyboard(deviceName='Forward2Xtrainfo_ID')
    
    # --- Initialize components for Routine "ID_Xtrainfo" ---
    Xtrainfo_ID = visual.TextStim(win=win, name='Xtrainfo_ID',
        text='The person you correctly identified did not earn the right to participate in this study and earn some money. He/She needs some financial assitance to take care of lunch, transportation, and other financial costs incurred as a reuslt of preparations to participate in this study. \nWhen you are ready to continue, please press the SPACE bar and follow the prompt. ',
        font='Arial',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    Forward2donate = keyboard.Keyboard(deviceName='Forward2donate')
    
    # --- Initialize components for Routine "Individual_donation" ---
    Request4donation = visual.TextStim(win=win, name='Request4donation',
        text='Please indicate how much you would like to donate to the person indentified as needing some financial help. Put your donation in the envelope at the end of this session.\n\nNOTE:\n\n\nPlease press the SPACE bar to continue\n',
        font='Arial',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    Forward2contribution = keyboard.Keyboard(deviceName='Forward2contribution')
    
    # --- Initialize components for Routine "last_stop_routine" ---
    key_resp = keyboard.Keyboard(deviceName='key_resp')
    option_0 = visual.TextStim(win=win, name='option_0',
        text='Press 0 - 0',
        font='Arial',
        pos=(0.0, -0.4), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-1.0);
    option_1 = visual.TextStim(win=win, name='option_1',
        text='Press 1 - 50',
        font='Arial',
        pos=(0.0, -0.3), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-2.0);
    option_2 = visual.TextStim(win=win, name='option_2',
        text='Press 2 - 100',
        font='Arial',
        pos=(0.0, -0.2), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-3.0);
    option_3 = visual.TextStim(win=win, name='option_3',
        text='Press 3 - 150',
        font='Arial',
        pos=(0.0, -0.1), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-4.0);
    option_4 = visual.TextStim(win=win, name='option_4',
        text='press 4 - 200',
        font='Arial',
        pos=(0.0, 0.0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-5.0);
    option_5 = visual.TextStim(win=win, name='option_5',
        text='press 5 - 250',
        font='Arial',
        pos=(0.0, 0.1), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-6.0);
    the_choice = visual.TextStim(win=win, name='the_choice',
        text='Please make a selection',
        font='Arial',
        pos=(0.0, 0.4), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-7.0);
    mouse = event.Mouse(win=win)
    x, y = [None, None]
    mouse.mouseClock = core.Clock()
    key_resp_2 = keyboard.Keyboard(deviceName='key_resp_2')
    
    # create some handy timers
    
    # global clock to track the time since experiment started
    if globalClock is None:
        # create a clock if not given one
        globalClock = core.Clock()
    if isinstance(globalClock, str):
        # if given a string, make a clock accoridng to it
        if globalClock == 'float':
            # get timestamps as a simple value
            globalClock = core.Clock(format='float')
        elif globalClock == 'iso':
            # get timestamps in ISO format
            globalClock = core.Clock(format='%Y-%m-%d_%H:%M:%S.%f%z')
        else:
            # get timestamps in a custom format
            globalClock = core.Clock(format=globalClock)
    if ioServer is not None:
        ioServer.syncClock(globalClock)
    logging.setDefaultClock(globalClock)
    # routine timer to track time remaining of each (possibly non-slip) routine
    routineTimer = core.Clock()
    win.flip()  # flip window to reset last flip timer
    # store the exact time the global clock started
    expInfo['expStart'] = data.getDateStr(
        format='%Y-%m-%d %Hh%M.%S.%f %z', fractionalSecondDigits=6
    )
    
    # --- Prepare to start Routine "Welcome_scr" ---
    # create an object to store info about Routine Welcome_scr
    Welcome_scr = data.Routine(
        name='Welcome_scr',
        components=[Welcome_screen, Forward2baseline_instruct],
    )
    Welcome_scr.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # create starting attributes for Forward2baseline_instruct
    Forward2baseline_instruct.keys = []
    Forward2baseline_instruct.rt = []
    _Forward2baseline_instruct_allKeys = []
    # store start times for Welcome_scr
    Welcome_scr.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    Welcome_scr.tStart = globalClock.getTime(format='float')
    Welcome_scr.status = STARTED
    thisExp.addData('Welcome_scr.started', Welcome_scr.tStart)
    Welcome_scr.maxDuration = None
    # keep track of which components have finished
    Welcome_scrComponents = Welcome_scr.components
    for thisComponent in Welcome_scr.components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "Welcome_scr" ---
    Welcome_scr.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *Welcome_screen* updates
        
        # if Welcome_screen is starting this frame...
        if Welcome_screen.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            Welcome_screen.frameNStart = frameN  # exact frame index
            Welcome_screen.tStart = t  # local t and not account for scr refresh
            Welcome_screen.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(Welcome_screen, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'Welcome_screen.started')
            # update status
            Welcome_screen.status = STARTED
            Welcome_screen.setAutoDraw(True)
        
        # if Welcome_screen is active this frame...
        if Welcome_screen.status == STARTED:
            # update params
            pass
        
        # *Forward2baseline_instruct* updates
        waitOnFlip = False
        
        # if Forward2baseline_instruct is starting this frame...
        if Forward2baseline_instruct.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            Forward2baseline_instruct.frameNStart = frameN  # exact frame index
            Forward2baseline_instruct.tStart = t  # local t and not account for scr refresh
            Forward2baseline_instruct.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(Forward2baseline_instruct, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'Forward2baseline_instruct.started')
            # update status
            Forward2baseline_instruct.status = STARTED
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(Forward2baseline_instruct.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(Forward2baseline_instruct.clearEvents, eventType='keyboard')  # clear events on next screen flip
        if Forward2baseline_instruct.status == STARTED and not waitOnFlip:
            theseKeys = Forward2baseline_instruct.getKeys(keyList=['space'], ignoreKeys=["escape"], waitRelease=False)
            _Forward2baseline_instruct_allKeys.extend(theseKeys)
            if len(_Forward2baseline_instruct_allKeys):
                Forward2baseline_instruct.keys = _Forward2baseline_instruct_allKeys[-1].name  # just the last key pressed
                Forward2baseline_instruct.rt = _Forward2baseline_instruct_allKeys[-1].rt
                Forward2baseline_instruct.duration = _Forward2baseline_instruct_allKeys[-1].duration
                # a response ends the routine
                continueRoutine = False
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer], 
                playbackComponents=[]
            )
            # skip the frame we paused on
            continue
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            Welcome_scr.forceEnded = routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in Welcome_scr.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "Welcome_scr" ---
    for thisComponent in Welcome_scr.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for Welcome_scr
    Welcome_scr.tStop = globalClock.getTime(format='float')
    Welcome_scr.tStopRefresh = tThisFlipGlobal
    thisExp.addData('Welcome_scr.stopped', Welcome_scr.tStop)
    # check responses
    if Forward2baseline_instruct.keys in ['', [], None]:  # No response was made
        Forward2baseline_instruct.keys = None
    thisExp.addData('Forward2baseline_instruct.keys',Forward2baseline_instruct.keys)
    if Forward2baseline_instruct.keys != None:  # we had a response
        thisExp.addData('Forward2baseline_instruct.rt', Forward2baseline_instruct.rt)
        thisExp.addData('Forward2baseline_instruct.duration', Forward2baseline_instruct.duration)
    thisExp.nextEntry()
    # the Routine "Welcome_scr" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # --- Prepare to start Routine "Instruction4baseline" ---
    # create an object to store info about Routine Instruction4baseline
    Instruction4baseline = data.Routine(
        name='Instruction4baseline',
        components=[Guide4baseline, Forward2baseline],
    )
    Instruction4baseline.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # create starting attributes for Forward2baseline
    Forward2baseline.keys = []
    Forward2baseline.rt = []
    _Forward2baseline_allKeys = []
    # store start times for Instruction4baseline
    Instruction4baseline.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    Instruction4baseline.tStart = globalClock.getTime(format='float')
    Instruction4baseline.status = STARTED
    thisExp.addData('Instruction4baseline.started', Instruction4baseline.tStart)
    Instruction4baseline.maxDuration = None
    # keep track of which components have finished
    Instruction4baselineComponents = Instruction4baseline.components
    for thisComponent in Instruction4baseline.components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "Instruction4baseline" ---
    Instruction4baseline.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *Guide4baseline* updates
        
        # if Guide4baseline is starting this frame...
        if Guide4baseline.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            Guide4baseline.frameNStart = frameN  # exact frame index
            Guide4baseline.tStart = t  # local t and not account for scr refresh
            Guide4baseline.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(Guide4baseline, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'Guide4baseline.started')
            # update status
            Guide4baseline.status = STARTED
            Guide4baseline.setAutoDraw(True)
        
        # if Guide4baseline is active this frame...
        if Guide4baseline.status == STARTED:
            # update params
            pass
        
        # *Forward2baseline* updates
        waitOnFlip = False
        
        # if Forward2baseline is starting this frame...
        if Forward2baseline.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            Forward2baseline.frameNStart = frameN  # exact frame index
            Forward2baseline.tStart = t  # local t and not account for scr refresh
            Forward2baseline.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(Forward2baseline, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'Forward2baseline.started')
            # update status
            Forward2baseline.status = STARTED
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(Forward2baseline.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(Forward2baseline.clearEvents, eventType='keyboard')  # clear events on next screen flip
        if Forward2baseline.status == STARTED and not waitOnFlip:
            theseKeys = Forward2baseline.getKeys(keyList=['space'], ignoreKeys=["escape"], waitRelease=False)
            _Forward2baseline_allKeys.extend(theseKeys)
            if len(_Forward2baseline_allKeys):
                Forward2baseline.keys = _Forward2baseline_allKeys[-1].name  # just the last key pressed
                Forward2baseline.rt = _Forward2baseline_allKeys[-1].rt
                Forward2baseline.duration = _Forward2baseline_allKeys[-1].duration
                # a response ends the routine
                continueRoutine = False
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer], 
                playbackComponents=[]
            )
            # skip the frame we paused on
            continue
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            Instruction4baseline.forceEnded = routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in Instruction4baseline.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "Instruction4baseline" ---
    for thisComponent in Instruction4baseline.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for Instruction4baseline
    Instruction4baseline.tStop = globalClock.getTime(format='float')
    Instruction4baseline.tStopRefresh = tThisFlipGlobal
    thisExp.addData('Instruction4baseline.stopped', Instruction4baseline.tStop)
    # check responses
    if Forward2baseline.keys in ['', [], None]:  # No response was made
        Forward2baseline.keys = None
    thisExp.addData('Forward2baseline.keys',Forward2baseline.keys)
    if Forward2baseline.keys != None:  # we had a response
        thisExp.addData('Forward2baseline.rt', Forward2baseline.rt)
        thisExp.addData('Forward2baseline.duration', Forward2baseline.duration)
    thisExp.nextEntry()
    # the Routine "Instruction4baseline" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # --- Prepare to start Routine "Baseline" ---
    # create an object to store info about Routine Baseline
    Baseline = data.Routine(
        name='Baseline',
        components=[Baseline_fixation],
    )
    Baseline.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # store start times for Baseline
    Baseline.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    Baseline.tStart = globalClock.getTime(format='float')
    Baseline.status = STARTED
    thisExp.addData('Baseline.started', Baseline.tStart)
    Baseline.maxDuration = None
    # keep track of which components have finished
    BaselineComponents = Baseline.components
    for thisComponent in Baseline.components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "Baseline" ---
    Baseline.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine and routineTimer.getTime() < 1.0:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *Baseline_fixation* updates
        
        # if Baseline_fixation is starting this frame...
        if Baseline_fixation.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            Baseline_fixation.frameNStart = frameN  # exact frame index
            Baseline_fixation.tStart = t  # local t and not account for scr refresh
            Baseline_fixation.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(Baseline_fixation, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'Baseline_fixation.started')
            # update status
            Baseline_fixation.status = STARTED
            Baseline_fixation.setAutoDraw(True)
        
        # if Baseline_fixation is active this frame...
        if Baseline_fixation.status == STARTED:
            # update params
            pass
        
        # if Baseline_fixation is stopping this frame...
        if Baseline_fixation.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > Baseline_fixation.tStartRefresh + 1.0-frameTolerance:
                # keep track of stop time/frame for later
                Baseline_fixation.tStop = t  # not accounting for scr refresh
                Baseline_fixation.tStopRefresh = tThisFlipGlobal  # on global time
                Baseline_fixation.frameNStop = frameN  # exact frame index
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'Baseline_fixation.stopped')
                # update status
                Baseline_fixation.status = FINISHED
                Baseline_fixation.setAutoDraw(False)
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer], 
                playbackComponents=[]
            )
            # skip the frame we paused on
            continue
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            Baseline.forceEnded = routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in Baseline.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "Baseline" ---
    for thisComponent in Baseline.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for Baseline
    Baseline.tStop = globalClock.getTime(format='float')
    Baseline.tStopRefresh = tThisFlipGlobal
    thisExp.addData('Baseline.stopped', Baseline.tStop)
    # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
    if Baseline.maxDurationReached:
        routineTimer.addTime(-Baseline.maxDuration)
    elif Baseline.forceEnded:
        routineTimer.reset()
    else:
        routineTimer.addTime(-1.000000)
    thisExp.nextEntry()
    
    # --- Prepare to start Routine "Instruct4tasks" ---
    # create an object to store info about Routine Instruct4tasks
    Instruct4tasks = data.Routine(
        name='Instruct4tasks',
        components=[Guide2tasks_instructions, Forward2task1],
    )
    Instruct4tasks.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # create starting attributes for Forward2task1
    Forward2task1.keys = []
    Forward2task1.rt = []
    _Forward2task1_allKeys = []
    # store start times for Instruct4tasks
    Instruct4tasks.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    Instruct4tasks.tStart = globalClock.getTime(format='float')
    Instruct4tasks.status = STARTED
    thisExp.addData('Instruct4tasks.started', Instruct4tasks.tStart)
    Instruct4tasks.maxDuration = None
    # keep track of which components have finished
    Instruct4tasksComponents = Instruct4tasks.components
    for thisComponent in Instruct4tasks.components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "Instruct4tasks" ---
    Instruct4tasks.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *Guide2tasks_instructions* updates
        
        # if Guide2tasks_instructions is starting this frame...
        if Guide2tasks_instructions.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            Guide2tasks_instructions.frameNStart = frameN  # exact frame index
            Guide2tasks_instructions.tStart = t  # local t and not account for scr refresh
            Guide2tasks_instructions.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(Guide2tasks_instructions, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'Guide2tasks_instructions.started')
            # update status
            Guide2tasks_instructions.status = STARTED
            Guide2tasks_instructions.setAutoDraw(True)
        
        # if Guide2tasks_instructions is active this frame...
        if Guide2tasks_instructions.status == STARTED:
            # update params
            pass
        
        # *Forward2task1* updates
        waitOnFlip = False
        
        # if Forward2task1 is starting this frame...
        if Forward2task1.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            Forward2task1.frameNStart = frameN  # exact frame index
            Forward2task1.tStart = t  # local t and not account for scr refresh
            Forward2task1.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(Forward2task1, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'Forward2task1.started')
            # update status
            Forward2task1.status = STARTED
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(Forward2task1.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(Forward2task1.clearEvents, eventType='keyboard')  # clear events on next screen flip
        if Forward2task1.status == STARTED and not waitOnFlip:
            theseKeys = Forward2task1.getKeys(keyList=['space'], ignoreKeys=["escape"], waitRelease=False)
            _Forward2task1_allKeys.extend(theseKeys)
            if len(_Forward2task1_allKeys):
                Forward2task1.keys = _Forward2task1_allKeys[-1].name  # just the last key pressed
                Forward2task1.rt = _Forward2task1_allKeys[-1].rt
                Forward2task1.duration = _Forward2task1_allKeys[-1].duration
                # a response ends the routine
                continueRoutine = False
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer], 
                playbackComponents=[]
            )
            # skip the frame we paused on
            continue
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            Instruct4tasks.forceEnded = routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in Instruct4tasks.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "Instruct4tasks" ---
    for thisComponent in Instruct4tasks.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for Instruct4tasks
    Instruct4tasks.tStop = globalClock.getTime(format='float')
    Instruct4tasks.tStopRefresh = tThisFlipGlobal
    thisExp.addData('Instruct4tasks.stopped', Instruct4tasks.tStop)
    # check responses
    if Forward2task1.keys in ['', [], None]:  # No response was made
        Forward2task1.keys = None
    thisExp.addData('Forward2task1.keys',Forward2task1.keys)
    if Forward2task1.keys != None:  # we had a response
        thisExp.addData('Forward2task1.rt', Forward2task1.rt)
        thisExp.addData('Forward2task1.duration', Forward2task1.duration)
    thisExp.nextEntry()
    # the Routine "Instruct4tasks" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # --- Prepare to start Routine "Task1" ---
    # create an object to store info about Routine Task1
    Task1 = data.Routine(
        name='Task1',
        components=[Intro_info_ID, Forward2Xtrainfo_ID],
    )
    Task1.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # create starting attributes for Forward2Xtrainfo_ID
    Forward2Xtrainfo_ID.keys = []
    Forward2Xtrainfo_ID.rt = []
    _Forward2Xtrainfo_ID_allKeys = []
    # store start times for Task1
    Task1.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    Task1.tStart = globalClock.getTime(format='float')
    Task1.status = STARTED
    thisExp.addData('Task1.started', Task1.tStart)
    Task1.maxDuration = None
    # keep track of which components have finished
    Task1Components = Task1.components
    for thisComponent in Task1.components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "Task1" ---
    Task1.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *Intro_info_ID* updates
        
        # if Intro_info_ID is starting this frame...
        if Intro_info_ID.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            Intro_info_ID.frameNStart = frameN  # exact frame index
            Intro_info_ID.tStart = t  # local t and not account for scr refresh
            Intro_info_ID.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(Intro_info_ID, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'Intro_info_ID.started')
            # update status
            Intro_info_ID.status = STARTED
            Intro_info_ID.setAutoDraw(True)
        
        # if Intro_info_ID is active this frame...
        if Intro_info_ID.status == STARTED:
            # update params
            pass
        
        # *Forward2Xtrainfo_ID* updates
        waitOnFlip = False
        
        # if Forward2Xtrainfo_ID is starting this frame...
        if Forward2Xtrainfo_ID.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            Forward2Xtrainfo_ID.frameNStart = frameN  # exact frame index
            Forward2Xtrainfo_ID.tStart = t  # local t and not account for scr refresh
            Forward2Xtrainfo_ID.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(Forward2Xtrainfo_ID, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'Forward2Xtrainfo_ID.started')
            # update status
            Forward2Xtrainfo_ID.status = STARTED
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(Forward2Xtrainfo_ID.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(Forward2Xtrainfo_ID.clearEvents, eventType='keyboard')  # clear events on next screen flip
        if Forward2Xtrainfo_ID.status == STARTED and not waitOnFlip:
            theseKeys = Forward2Xtrainfo_ID.getKeys(keyList=['space'], ignoreKeys=["escape"], waitRelease=False)
            _Forward2Xtrainfo_ID_allKeys.extend(theseKeys)
            if len(_Forward2Xtrainfo_ID_allKeys):
                Forward2Xtrainfo_ID.keys = _Forward2Xtrainfo_ID_allKeys[-1].name  # just the last key pressed
                Forward2Xtrainfo_ID.rt = _Forward2Xtrainfo_ID_allKeys[-1].rt
                Forward2Xtrainfo_ID.duration = _Forward2Xtrainfo_ID_allKeys[-1].duration
                # a response ends the routine
                continueRoutine = False
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer], 
                playbackComponents=[]
            )
            # skip the frame we paused on
            continue
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            Task1.forceEnded = routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in Task1.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "Task1" ---
    for thisComponent in Task1.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for Task1
    Task1.tStop = globalClock.getTime(format='float')
    Task1.tStopRefresh = tThisFlipGlobal
    thisExp.addData('Task1.stopped', Task1.tStop)
    # check responses
    if Forward2Xtrainfo_ID.keys in ['', [], None]:  # No response was made
        Forward2Xtrainfo_ID.keys = None
    thisExp.addData('Forward2Xtrainfo_ID.keys',Forward2Xtrainfo_ID.keys)
    if Forward2Xtrainfo_ID.keys != None:  # we had a response
        thisExp.addData('Forward2Xtrainfo_ID.rt', Forward2Xtrainfo_ID.rt)
        thisExp.addData('Forward2Xtrainfo_ID.duration', Forward2Xtrainfo_ID.duration)
    thisExp.nextEntry()
    # the Routine "Task1" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # --- Prepare to start Routine "ID_Xtrainfo" ---
    # create an object to store info about Routine ID_Xtrainfo
    ID_Xtrainfo = data.Routine(
        name='ID_Xtrainfo',
        components=[Xtrainfo_ID, Forward2donate],
    )
    ID_Xtrainfo.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # create starting attributes for Forward2donate
    Forward2donate.keys = []
    Forward2donate.rt = []
    _Forward2donate_allKeys = []
    # store start times for ID_Xtrainfo
    ID_Xtrainfo.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    ID_Xtrainfo.tStart = globalClock.getTime(format='float')
    ID_Xtrainfo.status = STARTED
    thisExp.addData('ID_Xtrainfo.started', ID_Xtrainfo.tStart)
    ID_Xtrainfo.maxDuration = None
    # keep track of which components have finished
    ID_XtrainfoComponents = ID_Xtrainfo.components
    for thisComponent in ID_Xtrainfo.components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "ID_Xtrainfo" ---
    ID_Xtrainfo.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *Xtrainfo_ID* updates
        
        # if Xtrainfo_ID is starting this frame...
        if Xtrainfo_ID.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            Xtrainfo_ID.frameNStart = frameN  # exact frame index
            Xtrainfo_ID.tStart = t  # local t and not account for scr refresh
            Xtrainfo_ID.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(Xtrainfo_ID, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'Xtrainfo_ID.started')
            # update status
            Xtrainfo_ID.status = STARTED
            Xtrainfo_ID.setAutoDraw(True)
        
        # if Xtrainfo_ID is active this frame...
        if Xtrainfo_ID.status == STARTED:
            # update params
            pass
        
        # *Forward2donate* updates
        waitOnFlip = False
        
        # if Forward2donate is starting this frame...
        if Forward2donate.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            Forward2donate.frameNStart = frameN  # exact frame index
            Forward2donate.tStart = t  # local t and not account for scr refresh
            Forward2donate.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(Forward2donate, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'Forward2donate.started')
            # update status
            Forward2donate.status = STARTED
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(Forward2donate.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(Forward2donate.clearEvents, eventType='keyboard')  # clear events on next screen flip
        if Forward2donate.status == STARTED and not waitOnFlip:
            theseKeys = Forward2donate.getKeys(keyList=['space'], ignoreKeys=["escape"], waitRelease=False)
            _Forward2donate_allKeys.extend(theseKeys)
            if len(_Forward2donate_allKeys):
                Forward2donate.keys = _Forward2donate_allKeys[-1].name  # just the last key pressed
                Forward2donate.rt = _Forward2donate_allKeys[-1].rt
                Forward2donate.duration = _Forward2donate_allKeys[-1].duration
                # a response ends the routine
                continueRoutine = False
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer], 
                playbackComponents=[]
            )
            # skip the frame we paused on
            continue
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            ID_Xtrainfo.forceEnded = routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in ID_Xtrainfo.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "ID_Xtrainfo" ---
    for thisComponent in ID_Xtrainfo.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for ID_Xtrainfo
    ID_Xtrainfo.tStop = globalClock.getTime(format='float')
    ID_Xtrainfo.tStopRefresh = tThisFlipGlobal
    thisExp.addData('ID_Xtrainfo.stopped', ID_Xtrainfo.tStop)
    # check responses
    if Forward2donate.keys in ['', [], None]:  # No response was made
        Forward2donate.keys = None
    thisExp.addData('Forward2donate.keys',Forward2donate.keys)
    if Forward2donate.keys != None:  # we had a response
        thisExp.addData('Forward2donate.rt', Forward2donate.rt)
        thisExp.addData('Forward2donate.duration', Forward2donate.duration)
    thisExp.nextEntry()
    # the Routine "ID_Xtrainfo" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # --- Prepare to start Routine "Individual_donation" ---
    # create an object to store info about Routine Individual_donation
    Individual_donation = data.Routine(
        name='Individual_donation',
        components=[Request4donation, Forward2contribution],
    )
    Individual_donation.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # create starting attributes for Forward2contribution
    Forward2contribution.keys = []
    Forward2contribution.rt = []
    _Forward2contribution_allKeys = []
    # store start times for Individual_donation
    Individual_donation.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    Individual_donation.tStart = globalClock.getTime(format='float')
    Individual_donation.status = STARTED
    thisExp.addData('Individual_donation.started', Individual_donation.tStart)
    Individual_donation.maxDuration = None
    # keep track of which components have finished
    Individual_donationComponents = Individual_donation.components
    for thisComponent in Individual_donation.components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "Individual_donation" ---
    Individual_donation.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *Request4donation* updates
        
        # if Request4donation is starting this frame...
        if Request4donation.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            Request4donation.frameNStart = frameN  # exact frame index
            Request4donation.tStart = t  # local t and not account for scr refresh
            Request4donation.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(Request4donation, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'Request4donation.started')
            # update status
            Request4donation.status = STARTED
            Request4donation.setAutoDraw(True)
        
        # if Request4donation is active this frame...
        if Request4donation.status == STARTED:
            # update params
            pass
        
        # *Forward2contribution* updates
        waitOnFlip = False
        
        # if Forward2contribution is starting this frame...
        if Forward2contribution.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            Forward2contribution.frameNStart = frameN  # exact frame index
            Forward2contribution.tStart = t  # local t and not account for scr refresh
            Forward2contribution.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(Forward2contribution, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'Forward2contribution.started')
            # update status
            Forward2contribution.status = STARTED
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(Forward2contribution.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(Forward2contribution.clearEvents, eventType='keyboard')  # clear events on next screen flip
        if Forward2contribution.status == STARTED and not waitOnFlip:
            theseKeys = Forward2contribution.getKeys(keyList=['space'], ignoreKeys=["escape"], waitRelease=False)
            _Forward2contribution_allKeys.extend(theseKeys)
            if len(_Forward2contribution_allKeys):
                Forward2contribution.keys = _Forward2contribution_allKeys[-1].name  # just the last key pressed
                Forward2contribution.rt = _Forward2contribution_allKeys[-1].rt
                Forward2contribution.duration = _Forward2contribution_allKeys[-1].duration
                # a response ends the routine
                continueRoutine = False
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer], 
                playbackComponents=[]
            )
            # skip the frame we paused on
            continue
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            Individual_donation.forceEnded = routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in Individual_donation.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "Individual_donation" ---
    for thisComponent in Individual_donation.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for Individual_donation
    Individual_donation.tStop = globalClock.getTime(format='float')
    Individual_donation.tStopRefresh = tThisFlipGlobal
    thisExp.addData('Individual_donation.stopped', Individual_donation.tStop)
    # check responses
    if Forward2contribution.keys in ['', [], None]:  # No response was made
        Forward2contribution.keys = None
    thisExp.addData('Forward2contribution.keys',Forward2contribution.keys)
    if Forward2contribution.keys != None:  # we had a response
        thisExp.addData('Forward2contribution.rt', Forward2contribution.rt)
        thisExp.addData('Forward2contribution.duration', Forward2contribution.duration)
    thisExp.nextEntry()
    # the Routine "Individual_donation" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # --- Prepare to start Routine "last_stop_routine" ---
    # create an object to store info about Routine last_stop_routine
    last_stop_routine = data.Routine(
        name='last_stop_routine',
        components=[key_resp, option_0, option_1, option_2, option_3, option_4, option_5, the_choice, mouse, key_resp_2],
    )
    last_stop_routine.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # create starting attributes for key_resp
    key_resp.keys = []
    key_resp.rt = []
    _key_resp_allKeys = []
    # setup some python lists for storing info about the mouse
    mouse.x = []
    mouse.y = []
    mouse.leftButton = []
    mouse.midButton = []
    mouse.rightButton = []
    mouse.time = []
    mouse.clicked_name = []
    gotValidClick = False  # until a click is received
    # Run 'Begin Routine' code from code
    win.mouseVisible= True
    
    def checkTheMouse():
        if(option_0.contains(mouse)):
            the_choice.text =   'You chose to give 0 NAIRA. To confirm click the space bar'
        if(option_1.contains(mouse)):
            the_choice.text =    'You chose to give 50 NAIRA. To confirm click the space bar'
         
        if(option_2.contains(mouse)):
            the_choice.text =   'You chose to give 100 NAIRA. To confirm click the space bar'
          
        if(option_3.contains(mouse)):
            the_choice.text =   'You chose to give 150 NAIRA. To confirm click the space bar'
            
        if(option_4.contains(mouse)):
            the_choice.text =    'You chose to give 200 NAIRA To confirm click the space bar' 
          
        if(option_5.contains(mouse)):
            the_choice.text =    'You chose to give 250 NAIRA. To confirm click the space bar'
            
        
    def checkTheKeyboard():
        if(keys_this_frame[-1].name=='0'):
            the_choice.text =    'You chose to give 0 NAIRA. To confirm click the space bar'
        if(keys_this_frame[-1].name=='1'):
            the_choice.text =    'You chose to give 50 NAIRA. To confirm click the space bar'
        if(keys_this_frame[-1].name=='2'):
            the_choice.text =    'You chose to give 100 NAIRA. To confirm click the space bar'    
        if(keys_this_frame[-1].name=='3'):
            the_choice.text =    'You chose to give 150 NAIRA. To confirm click the space bar'
        if(keys_this_frame[-1].name=='4'):
            the_choice.text =    'You chose to give 200 NAIRA. To confirm click the space bar'
        if(keys_this_frame[-1].name=='5'):
            the_choice.text =    'You chose to give 250 NAIRA. To confirm click the space bar'
        win.flip()
          
    
    # create starting attributes for key_resp_2
    key_resp_2.keys = []
    key_resp_2.rt = []
    _key_resp_2_allKeys = []
    # store start times for last_stop_routine
    last_stop_routine.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    last_stop_routine.tStart = globalClock.getTime(format='float')
    last_stop_routine.status = STARTED
    thisExp.addData('last_stop_routine.started', last_stop_routine.tStart)
    last_stop_routine.maxDuration = None
    # keep track of which components have finished
    last_stop_routineComponents = last_stop_routine.components
    for thisComponent in last_stop_routine.components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "last_stop_routine" ---
    last_stop_routine.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *key_resp* updates
        waitOnFlip = False
        
        # if key_resp is starting this frame...
        if key_resp.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            key_resp.frameNStart = frameN  # exact frame index
            key_resp.tStart = t  # local t and not account for scr refresh
            key_resp.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(key_resp, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'key_resp.started')
            # update status
            key_resp.status = STARTED
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(key_resp.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(key_resp.clearEvents, eventType='keyboard')  # clear events on next screen flip
        if key_resp.status == STARTED and not waitOnFlip:
            theseKeys = key_resp.getKeys(keyList=['0','1','2','3','4','5'], ignoreKeys=["escape"], waitRelease=False)
            _key_resp_allKeys.extend(theseKeys)
            if len(_key_resp_allKeys):
                key_resp.keys = _key_resp_allKeys[-1].name  # just the last key pressed
                key_resp.rt = _key_resp_allKeys[-1].rt
                key_resp.duration = _key_resp_allKeys[-1].duration
        
        # *option_0* updates
        
        # if option_0 is starting this frame...
        if option_0.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            option_0.frameNStart = frameN  # exact frame index
            option_0.tStart = t  # local t and not account for scr refresh
            option_0.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(option_0, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'option_0.started')
            # update status
            option_0.status = STARTED
            option_0.setAutoDraw(True)
        
        # if option_0 is active this frame...
        if option_0.status == STARTED:
            # update params
            pass
        
        # *option_1* updates
        
        # if option_1 is starting this frame...
        if option_1.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            option_1.frameNStart = frameN  # exact frame index
            option_1.tStart = t  # local t and not account for scr refresh
            option_1.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(option_1, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'option_1.started')
            # update status
            option_1.status = STARTED
            option_1.setAutoDraw(True)
        
        # if option_1 is active this frame...
        if option_1.status == STARTED:
            # update params
            pass
        
        # *option_2* updates
        
        # if option_2 is starting this frame...
        if option_2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            option_2.frameNStart = frameN  # exact frame index
            option_2.tStart = t  # local t and not account for scr refresh
            option_2.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(option_2, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'option_2.started')
            # update status
            option_2.status = STARTED
            option_2.setAutoDraw(True)
        
        # if option_2 is active this frame...
        if option_2.status == STARTED:
            # update params
            pass
        
        # *option_3* updates
        
        # if option_3 is starting this frame...
        if option_3.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            option_3.frameNStart = frameN  # exact frame index
            option_3.tStart = t  # local t and not account for scr refresh
            option_3.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(option_3, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'option_3.started')
            # update status
            option_3.status = STARTED
            option_3.setAutoDraw(True)
        
        # if option_3 is active this frame...
        if option_3.status == STARTED:
            # update params
            pass
        
        # *option_4* updates
        
        # if option_4 is starting this frame...
        if option_4.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            option_4.frameNStart = frameN  # exact frame index
            option_4.tStart = t  # local t and not account for scr refresh
            option_4.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(option_4, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'option_4.started')
            # update status
            option_4.status = STARTED
            option_4.setAutoDraw(True)
        
        # if option_4 is active this frame...
        if option_4.status == STARTED:
            # update params
            pass
        
        # *option_5* updates
        
        # if option_5 is starting this frame...
        if option_5.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            option_5.frameNStart = frameN  # exact frame index
            option_5.tStart = t  # local t and not account for scr refresh
            option_5.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(option_5, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'option_5.started')
            # update status
            option_5.status = STARTED
            option_5.setAutoDraw(True)
        
        # if option_5 is active this frame...
        if option_5.status == STARTED:
            # update params
            pass
        
        # *the_choice* updates
        
        # if the_choice is starting this frame...
        if the_choice.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            the_choice.frameNStart = frameN  # exact frame index
            the_choice.tStart = t  # local t and not account for scr refresh
            the_choice.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(the_choice, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'the_choice.started')
            # update status
            the_choice.status = STARTED
            the_choice.setAutoDraw(True)
        
        # if the_choice is active this frame...
        if the_choice.status == STARTED:
            # update params
            pass
        # *mouse* updates
        
        # if mouse is starting this frame...
        if mouse.status == NOT_STARTED and t >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            mouse.frameNStart = frameN  # exact frame index
            mouse.tStart = t  # local t and not account for scr refresh
            mouse.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(mouse, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.addData('mouse.started', t)
            # update status
            mouse.status = STARTED
            mouse.mouseClock.reset()
            prevButtonState = [0, 0, 0]  # if now button is down we will treat as 'new' click
        if mouse.status == STARTED:  # only update if started and not finished!
            buttons = mouse.getPressed()
            if buttons != prevButtonState:  # button state changed?
                prevButtonState = buttons
                if sum(buttons) > 0:  # state changed to a new click
                    # check if the mouse was inside our 'clickable' objects
                    gotValidClick = False
                    clickableList = environmenttools.getFromNames([option_0,option_1,option_2,option_3,option_4,option_5], namespace=locals())
                    for obj in clickableList:
                        # is this object clicked on?
                        if obj.contains(mouse):
                            gotValidClick = True
                            mouse.clicked_name.append(obj.name)
                    if not gotValidClick:
                        mouse.clicked_name.append(None)
                    x, y = mouse.getPos()
                    mouse.x.append(x)
                    mouse.y.append(y)
                    buttons = mouse.getPressed()
                    mouse.leftButton.append(buttons[0])
                    mouse.midButton.append(buttons[1])
                    mouse.rightButton.append(buttons[2])
                    mouse.time.append(mouse.mouseClock.getTime())
        # Run 'Each Frame' code from code
        
        keys_this_frame = _key_resp_allKeys
        
        if sum(mouse.getPressed()):
            checkTheMouse()
            the_choice.draw()
            core.wait(0.2)
        
        if len(keys_this_frame):
            checkTheKeyboard()
            the_choice.draw()
            core.wait(0.2)
            keys_this_frame = None
            
        
        
        
        
            
        
        
        
        # *key_resp_2* updates
        waitOnFlip = False
        
        # if key_resp_2 is starting this frame...
        if key_resp_2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            key_resp_2.frameNStart = frameN  # exact frame index
            key_resp_2.tStart = t  # local t and not account for scr refresh
            key_resp_2.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(key_resp_2, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'key_resp_2.started')
            # update status
            key_resp_2.status = STARTED
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(key_resp_2.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(key_resp_2.clearEvents, eventType='keyboard')  # clear events on next screen flip
        if key_resp_2.status == STARTED and not waitOnFlip:
            theseKeys = key_resp_2.getKeys(keyList=['space'], ignoreKeys=["escape"], waitRelease=False)
            _key_resp_2_allKeys.extend(theseKeys)
            if len(_key_resp_2_allKeys):
                key_resp_2.keys = _key_resp_2_allKeys[-1].name  # just the last key pressed
                key_resp_2.rt = _key_resp_2_allKeys[-1].rt
                key_resp_2.duration = _key_resp_2_allKeys[-1].duration
                # a response ends the routine
                continueRoutine = False
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer], 
                playbackComponents=[]
            )
            # skip the frame we paused on
            continue
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            last_stop_routine.forceEnded = routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in last_stop_routine.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "last_stop_routine" ---
    for thisComponent in last_stop_routine.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for last_stop_routine
    last_stop_routine.tStop = globalClock.getTime(format='float')
    last_stop_routine.tStopRefresh = tThisFlipGlobal
    thisExp.addData('last_stop_routine.stopped', last_stop_routine.tStop)
    # store data for thisExp (ExperimentHandler)
    thisExp.addData('mouse.x', mouse.x)
    thisExp.addData('mouse.y', mouse.y)
    thisExp.addData('mouse.leftButton', mouse.leftButton)
    thisExp.addData('mouse.midButton', mouse.midButton)
    thisExp.addData('mouse.rightButton', mouse.rightButton)
    thisExp.addData('mouse.time', mouse.time)
    thisExp.addData('mouse.clicked_name', mouse.clicked_name)
    # check responses
    if key_resp_2.keys in ['', [], None]:  # No response was made
        key_resp_2.keys = None
    thisExp.addData('key_resp_2.keys',key_resp_2.keys)
    if key_resp_2.keys != None:  # we had a response
        thisExp.addData('key_resp_2.rt', key_resp_2.rt)
        thisExp.addData('key_resp_2.duration', key_resp_2.duration)
    thisExp.nextEntry()
    # the Routine "last_stop_routine" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # mark experiment as finished
    endExperiment(thisExp, win=win)


def saveData(thisExp):
    """
    Save data from this experiment
    
    Parameters
    ==========
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    """
    filename = thisExp.dataFileName
    # these shouldn't be strictly necessary (should auto-save)
    thisExp.saveAsWideText(filename + '.csv', delim='auto')
    thisExp.saveAsPickle(filename)


def endExperiment(thisExp, win=None):
    """
    End this experiment, performing final shut down operations.
    
    This function does NOT close the window or end the Python process - use `quit` for this.
    
    Parameters
    ==========
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    win : psychopy.visual.Window
        Window for this experiment.
    """
    if win is not None:
        # remove autodraw from all current components
        win.clearAutoDraw()
        # Flip one final time so any remaining win.callOnFlip() 
        # and win.timeOnFlip() tasks get executed
        win.flip()
    # return console logger level to WARNING
    logging.console.setLevel(logging.WARNING)
    # mark experiment handler as finished
    thisExp.status = FINISHED
    logging.flush()


def quit(thisExp, win=None, thisSession=None):
    """
    Fully quit, closing the window and ending the Python process.
    
    Parameters
    ==========
    win : psychopy.visual.Window
        Window to close.
    thisSession : psychopy.session.Session or None
        Handle of the Session object this experiment is being run from, if any.
    """
    thisExp.abort()  # or data files will save again on exit
    # make sure everything is closed down
    if win is not None:
        # Flip one final time so any remaining win.callOnFlip() 
        # and win.timeOnFlip() tasks get executed before quitting
        win.flip()
        win.close()
    logging.flush()
    if thisSession is not None:
        thisSession.stop()
    # terminate Python process
    core.quit()


# if running this experiment as a script...
if __name__ == '__main__':
    # call all functions in order
    expInfo = showExpInfoDlg(expInfo=expInfo)
    thisExp = setupData(expInfo=expInfo)
    logFile = setupLogging(filename=thisExp.dataFileName)
    win = setupWindow(expInfo=expInfo)
    setupDevices(expInfo=expInfo, thisExp=thisExp, win=win)
    run(
        expInfo=expInfo, 
        thisExp=thisExp, 
        win=win,
        globalClock='float'
    )
    saveData(thisExp=thisExp)
    quit(thisExp=thisExp, win=win)
