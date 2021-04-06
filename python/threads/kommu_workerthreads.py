#!/usr/bin/env python
# coding: utf-8
#
# Title: Kommunikationssysteme: PythonThreads - Producer / Consumer
# Author: Sidney Göhler
#
# Description: Dieses Skript simuliert eine bestimmte Anzahl an Producer / Consumer, 
#              welche parallel Daten in eine Queue schreiben bzw. auslesen. 
#              Ist die Queue voll werden Producer pausiert und ggf. später wieder
#              aktiviert. Ist die Queue leer werden Consumer pausiert und ggf. wieder
#              aktiviert.
#              Die Queue aus dem Modul sind threadsicher, weswegen die Lcoks
#              nicht nochmal implemntiert werden müssen 
#               https://docs.python.org/3/library/queue.html
#              Für die Überwachung der Producer / Consumer Threads wurde eine weitere
#              Klasse implementiert, welche über ein Event und eine weitere Queue
#              erfährt, welcher Thread schläft, wobei man das sicher auch eleganter
#              machen kann.
#              Bin mir nicht mal sicher, ob man die zweite Queue für
#              diese Fall überhaupt benötigt, da man den aktiv/pause status auch
#              einfach abfragen kann.
#              In erster Linie wird die zweite Queue genutzt um zu erkennen, wenn
#              mehr als ein Thread schlaeft.
#              Die zweite Queue könnte aber auch als Möglichkeit für einen callback
#              verwendet werden.
#
#
#!/usr/bin/env python
# coding: utf-8

# In[1]:


from queue import Queue
from threading import Thread, Event
import time
import random


# In[2]:


class PCManager(Thread):
    def __init__(self):
        super().__init__()
        self.dataQueue = None
        self.producerList = []
        self.producerActive = 0
        self.consumerList = []
        self.consumerActive = 0
        self.threadProducerDoneEvent = Event()
        self.threadSleepEvent = Event()
        self.messageQueue = Queue() # message Queue für thread callbacks

    def createProducer(self):
        for i in range(nbProducer):
            print(f'creating producer ({i+1}/{nb_ProducerConsumer}) with id: {i}\n')
            self.producerList.append(Producer(self, i, self.dataQueue, self.threadProducerDoneEvent))

    def createConsumer(self):
        for i in range(nbConsumer):
            print(f'creating consumer ({i+1}/{nb_ProducerConsumer}) with id: {i}\n')
            self.consumerList.append(Consumer(self, i, self.dataQueue,self.threadProducerDoneEvent))

    #Producer und Consumer gemeinsam erstellen
    def createPC(self, nb_ProducerConsumer):
        for i in range(nb_ProducerConsumer):
            print(f'creating producer ({i+1}/{nb_ProducerConsumer}) with id: {i}\n')
            self.producerList.append(Producer(self, i, self.dataQueue, self.threadProducerDoneEvent))
            print(f'creating consumer ({i+1}/{nb_ProducerConsumer}) with id: {i}\n')
            self.consumerList.append(Consumer(self, i, self.dataQueue, self.threadProducerDoneEvent))

    def setDataQueue(self, dataQueue):
        self.dataQueue = dataQueue  #Dataqueue setzen

    def runProducer(self):
        # starte die producer
        if not dataQueue is None:
            self.producerActive = len(self.producerList)
            for producer in self.producerList:
                producer.start()

    def runConsumer(self):
        # starte die consumer
        if not dataQueue is None:
            self.consumerActive = len(self.consumerList)
            for consumer in self.consumerList:
                consumer.start()

    # alle threads starten
    def runAll(self):
        if not dataQueue is None:
            self.consumerActive = len(self.consumerList)
            self.producerActive = len(self.producerList)
            for p, c in zip(self.producerList, self.consumerList):
                p.start()
                c.start()

    def pauseById(self, id):
        for p, c in zip(self.producerList, self.consumerList):
            if p.identifier == id:
                p.pause()
            elif c.identifier == id:
                c.pause()

    def pauseAll(self):
        for p, c in zip(self.producerList, self.consumerList):
            p.pause()
            c.pause()

    def resumeById(self, id):
        for p, c in zip(self.producerList, self.consumerList):
            if p.identifier == id and not self.dataQueue.full():
                p.resume()
            elif c.identifier == id and not self.dataQueue.empty():
                c.resume()

    def resumeAll(self):
        for p, c in zip(self.producerList, self.consumerList):
            p.resume()
            c.resume()

    #Override
    def run(self):
        while 1:
            self.threadSleepEvent.wait()    # warte auf sleepevent von einem producer/consumer
            threadId = self.messageQueue.get()  # gib mir die Id aus der vom thread beschriebenen queue
            
            # welcher thread ist eingeschlafen und darf er schon wieder aufwachen
            for p, c in zip(self.producerList, self.consumerList):
                if p.identifier == threadId and p.isPaused() and not self.dataQueue.full():
                    p.resume()

                if c.identifier == threadId and c.isPaused() and not self.dataQueue.empty():
                    c.resume()

            self.messageQueue.task_done()

            # wenn die queue noch nicht leer ist schlaeft noch ein anderer Thread
            if self.messageQueue.empty():
                self.threadSleepEvent.clear()

# In[3]:


class Producer(Thread):
    def __init__(self, PCManager, identifier, dataQueue, doneEvent):
        super().__init__()
        self.manager = PCManager
        self.identifier = identifier #id(self)
        self.runThread = Event() # aktivflag
        self.runThread.set() # thread ist aktiviert
        self.dataQueue = dataQueue # Datenqueue
        self.done = doneEvent #producer fertig Event

    #Override
    def run(self):
        while 1:
            self.runThread.wait() # producer pausiert?
            self.start = time.time()
            if not self.dataQueue.full(): # Datenqueue nicht voll?
                item = random.randint(0,65535)
                randi = random.randint(3,8)
                try:
                    self.done.clear() # neuer Schreibvorgang
                    time.sleep(randi) # warte zwischen 1 und 8 Sekunden
                    self.dataQueue.put(item) # schreibe Wert
                    self.end = time.time()
                    print(f'(producer) {self.identifier}: PUT value {hex(item)}, took {(self.end - self.start):.3f} seconds!\n' +
                          f'Queue: {self.dataQueue.qsize()}/{self.dataQueue.maxsize}\n')

                    self.done.set() # producer fertig

                except Exception as e:
                    print(f'{type(e).__name__}: {e.args}')

            else:
                self.pause() # thread pausieren
                # callback
                self.manager.messageQueue.put(self.identifier) # die messagequeue im manager mit der id beschreiben
                self.manager.threadSleepEvent.set() # das sleepevent flag setzen

    def pause(self):
        self.startpause = time.time()
        self.manager.producerActive -= 1
        self.runThread.clear() # thread pausieren
        print(f'(Producer) {self.identifier}: PAUSED!\n{self.manager.producerActive} producer active.\n')

    def isPaused(self):
        return not self.runThread.is_set()

    def resume(self):      
        self.manager.producerActive += 1
        self.runThread.set() # thread fortsetzen
        self.endpause = time.time()
        print(f'(Producer) {self.identifier}: RESUMED! slept for {(self.endpause - self.startpause):.3f} seconds!\n{self.manager.producerActive} producer active.\n')


# In[4]:


class Consumer(Thread):
    def __init__(self, PCManager, identifier, dataQueue, doneEvent):
        super().__init__()
        self.manager = PCManager
        self.identifier = identifier #id(self)
        self.runThread = Event() # aktivflag
        self.runThread.set() # thread ist aktiviert
        self.dataQueue = dataQueue
        self.producerDone = doneEvent

    #Override
    def run(self):
        while 1:
            self.runThread.wait() # Consumer pausiert ?
            self.producerDone.wait() # warte auf Producer
            self.start = time.time()
            if not self.dataQueue.empty(): # Datenqueue nicht leer?
                randi = random.randint(5,11)
                try:
                    time.sleep(randi) # warte zwischen 3 und 12 Sekunden
                    item = self.dataQueue.get() # lese Wert
                    self.end = time.time()
                    print(f'(consumer) {self.identifier}: GET value {hex(item)}, took {(self.end - self.start):.3f} seconds!\n' +
                          f'Queue: {self.dataQueue.qsize()}/{self.dataQueue.maxsize}\n')
                    dataQueue.task_done() # get fertig

                except Exception as e:
                    print(f'{type(e).__name__}: {e.args}')

            else:
                self.pause() # thread pausieren
                self.manager.messageQueue.put(self.identifier) # die messagequeue im manager mit der id beschreiben
                self.manager.threadSleepEvent.set() # das sleepevent flag setzen

    def pause(self):
        self.startpause = time.time()
        self.manager.consumerActive -= 1
        self.runThread.clear() # thread pausieren
        print(f'(consumer) {self.identifier}: PAUSED\n{self.manager.consumerActive} consumer active.\n')

    def isPaused(self):
        return not self.runThread.is_set()

    def resume(self):
        self.runThread.set() # thread fortsetzen
        self.manager.consumerActive += 1
        self.endpause = time.time()
        print(f'(consumer) {self.identifier}: RESUMED! slept for {(self.endpause - self.startpause):.3f} seconds!\n{self.manager.consumerActive} consumer active.\n')



# In[5]:


if __name__ == "__main__":
    # ertelle eine Datenqueue
    BUFF_SIZE = 10 # groesse der queue
    dataQueue = Queue(BUFF_SIZE)

    # erstelle eine "zufaellige" Anzahl and consumer und producer

    print('#####################################')
    print('### Create Threads')
    print('#####################################\n')

    pcm = PCManager()
    pcm.setDataQueue(dataQueue)
    pcm.createPC(6)

    print('#####################################')
    print('### Start Threads')
    print('#####################################\n')

    pcm.runAll() # starte alle producer/consumer
    pcm.start() # starte manager

