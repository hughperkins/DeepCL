import gym
from pgi.repository import GLib
import dbus.service
import dbus.mainloop.glib

class DbusEnv(dbus.service.Object):
    dbus_iface = 'gym.cart.env'

    __env = gym.make('CartPole-v0')
    __env.reset()
    __env = __env.unwrapped
    
    print(__env.action_space)
    print(__env.observation_space)
    print(__env.observation_space.high)
    print(__env.observation_space.low)
    
    @dbus.service.method(dbus_interface=dbus_iface)
    def render(self):
        mode = 'human'
        close = False
        return self.__env.render(mode, close)
    
    @dbus.service.method(dbus_interface=dbus_iface, in_signature='i', out_signature='addb')
    def step(self, action):
        ob, reward, done, _ = self.__env.step(action)
        return ob, reward, done
    
    @dbus.service.method(dbus_interface=dbus_iface, out_signature='ad')
    def reset(self):
        return self.__env.reset()        

if __name__ == '__main__':
    dbus.mainloop.glib.DBusGMainLoop(set_as_default=True)    

    session_bus = dbus.SessionBus()
    name = dbus.service.BusName(DbusEnv.dbus_iface + '.service', session_bus)
    object = DbusEnv(session_bus, '/' + DbusEnv.dbus_iface.replace('.', '/'))

    mainloop = GLib.MainLoop()
    print "Running example service."
    mainloop.run()
    
