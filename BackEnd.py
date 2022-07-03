import pyttsx3

engine = pyttsx3.init()

# set some parameters
rate = engine.getProperty('rate')   # the voice rate
engine.setProperty('rate', 170)
voices = engine.getProperty('voices')  # getting details of current voice
engine.setProperty('voice', voices[1].id)  # set a female voice

# change the volume
def setVolume(newVolume):
    engine.setProperty('volume', newVolume)


# convert text to speech
def Speak(speak_text):
    engine.say(speak_text)
    engine.runAndWait()
    engine.stop()

