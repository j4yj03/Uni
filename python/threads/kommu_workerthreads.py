#!/usr/bin/env python
# coding: utf-8

# In[1]:


from queue import Queue
from threading import Thread, Lock, Event
from time import sleep
import random


# In[5]:


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

# In[2]:


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
            self.runThread.wait()
            if not self.dataQueue.full(): # Datenqueue nicht voll?
                item = random.randint(1,64)
                randi = random.randint(1,8)
                try:
                    self.done.clear() # neuer Schreibvorgang
                    sleep(randi) # warte zwischen 1 und 8 Sekunden
                    self.dataQueue.put(item) # schreibe Wert
                    print(f'(producer) {self.identifier}: PUT value {item}, took {randi} seconds!\n' +
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
        print(f'(Producer) {self.identifier}: PAUSED!\n{self.manager.producerActive} producer active.\n')
        self.manager.producerActive -= 1
        self.runThread.clear() # thread pausieren

    def isPaused(self):
        return not self.runThread.is_set()

    def resume(self):
        print(f'(Producer) {self.identifier}: RESUMED!\n{self.manager.producerActive} producer active.\n')
        self.manager.producerActive += 1
        self.runThread.set() # thread fortsetzen


# In[3]:


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
            if not self.dataQueue.empty(): # Datenqueue nicht leer?
                randi = random.randint(1,8)
                try:
                    sleep(randi) # warte zwischen 1 und 8 Sekunden
                    item = self.dataQueue.get() # lese Wert
                    print(f'(consumer) {self.identifier}: GET value {item}, took {randi} seconds!\n' +
                          f'Queue: {self.dataQueue.qsize()}/{self.dataQueue.maxsize}\n')
                    dataQueue.task_done() # get fertig

                except Exception as e:
                    print(f'{type(e).__name__}: {e.args}')

            else:
                self.pause() # thread pausieren
                self.manager.messageQueue.put(self.identifier) # die messagequeue im manager mit der id beschreiben
                self.manager.threadSleepEvent.set() # das sleepevent flag setzen

    def pause(self):
        print(f'(consumer) {self.identifier}: PAUSED\n{self.manager.consumerActive} consumer active.\n')
        self.manager.consumerActive -= 1
        self.runThread.clear() # thread pausieren

    def isPaused(self):
        return not self.runThread.is_set()

    def resume(self):
        self.runThread.set() # thread fortsetzen
        self.manager.consumerActive += 1
        print(f'(consumer) {self.identifier}: RESUMED!\n{self.manager.consumerActive} consumer active.\n')



# In[4]:


if __name__ == "__main__":
    # ertelle eine Datenqueue
    BUFF_SIZE = 3 # groesse der queue
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
