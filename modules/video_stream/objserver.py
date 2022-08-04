
class Subject(object):
    def __init__(self):
        self.__dict__['state'] = 0
        self.__dict__['observers'] = []

    def __setattr__(self, name, value):
        self.__dict__[name] = value
        if name == 'state':
            self.notify_observers()

    def register(self, observer):
        if observer not in self.observers:
            self.observers.append(observer)

    def deregister(self, observer):
        self.observers.remove(observer)

    def notify_observers(self):
        for observer in self.observers:
            observer.update()

class Observer(object):
    def update(self):
        raise NotImplementedError("update() is not implemented.")