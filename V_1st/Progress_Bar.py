import time

class Progress_Bar():

    def __init__(self,Total,Name,bar_type=0,loop_type=0):
        self.Name=Name
        self.bar_length=10
        self.progress_length=Total
        self.progress_current=0
        self.bar_type=bar_type
        self.pre=[' |','  ','  ','  ']
        self.suf=['| ','  ','  ','  ']
        self.fill=['█','▣','◉','▶']
        self.emp=['∙','▢','◯','▷']
        self.loop_type=loop_type
        self.loop=[['◤','◥','◢','◣'],\
                   ['⣾','⣷', '⣯', '⣟', '⡿', '⢿', '⣻', '⣽'],\
                   ['-','\\','|','/']]
        self.show_string=''
        self.start_time=time.time()
        self.step_time=0.0
        self.end_time=0.0

    def show(self,i=None):
        if i is not None:
            self.progress_current=i
        else:
            self.progress_current+=1
        per=int(self.progress_current*100/self.progress_length)
        n=int(self.progress_current*self.bar_length/self.progress_length)
        rn=int(self.bar_length-n)
        ln=n%len(self.loop[self.loop_type])

        self.step_time=time.time()
        argtime=(self.step_time-self.start_time)/self.progress_current
        #show_string=self.Name+': '+self.pre[self.bar_type]+self.fill[self.bar_type]*n+self.emp[self.bar_type]*rn+self.suf[self.bar_type]+str(per)+'%'+self.loop[self.loop_type][ln]
        self.show_string=self.Name+': '+self.pre[self.bar_type]+self.fill[self.bar_type]*n+self.emp[self.bar_type]*rn+self.suf[self.bar_type]+str(self.progress_current)+' / '+str(self.progress_length)+' ArgTime: '+str(argtime)+' Seconds'
        print(self.show_string, end='\r', flush=True)
        time.sleep(0.01)

    def close(self):
        print(self.show_string)
        self.end_time=time.time()
        alltime=self.end_time-self.start_time
        close_string='\n'+'*******'+self.Name+' '+'Done! '+'AllTime: '+str(alltime)+' Seconds!'+'*******'
        print(close_string)
        self.progress_current=0
        time.sleep(1)
