import re
import pickle
import sys,os
import matplotlib.pyplot as plt
import numpy as np

DEBUG = 0

def debug(text):
    sys.stdout.write(text)
    sys.stdout.flush()

class Trace:
    def __init__(self):
        self.pc_list = dict()
        self.addr_list = dict()
        self.delta_list = dict()
        self.prev_addr = -1
        self.num_loads = 0
        self.benchmark = ""

    def add(self, addr, pc):
        self.num_loads += 1

        if not self.prev_addr == -1:
            delta = addr - self.prev_addr
            self.add_to_dict(self.delta_list, delta)
        self.add_to_dict(self.pc_list, pc)
        self.add_to_dict(self.addr_list, addr)
        self.prev_addr = addr

    def add_to_dict(self, dictionary, value):
        if value in dictionary:
            dictionary[value] += 1
        else:
            dictionary[value] = 1

    def crawl(self, filename):
        self.benchmark = filename[:filename.index(".")]
        pattern = re.compile("\d+ \d+")

        debug("Crawling " + filename + "...\n")
        with open(filename, "r") as f:
            for line in f:
                if pattern.match(line.strip()):
                    addr, pc = [int(s) for s in line.strip().split(" ")]
                    self.add(addr, pc)
    def save(self, filename):
        debug("Saving trace to file " + filename +"!\n")
        with open(filename, "wb") as f:
            pickle.dump(self, f, pickle.HIGHEST_PROTOCOL)

    def load(self, filename):
        debug("Loading trace from file " + filename + "!\n");
        with open(filename, "rb") as f:
            dump = pickle.load(f)
            self.pc_list = dump.pc_list
            self.addr_list = dump.addr_list
            self.delta_list = dump.delta_list
            self.prev_addr = dump.prev_addr
            self.benchmark = dump.benchmark
            self.num_loads = dump.num_loads

    def needed_for_coverage(self, dictionary, coverage=0.50):
        goal = self.num_loads*0.50
        current = 0
        num_needed = 0

        for key in sorted(dictionary.keys(), key=lambda x: dictionary[x], reverse=True):
            current += dictionary[key]
            num_needed += 1
            if current >= goal:
                break
        if current < goal:
            raise Exception("needed_for_coverage broken")
        return num_needed

    def coverage_from_num(self, dictionary, num):
        num = min(num, len(dictionary.keys()))

        current = 0
        for key in sorted(dictionary.keys(), key=lambda x: dictionary[x], reverse=True)[:num]:
            current += dictionary[key]

        return float(current)/self.num_loads



    def plot_frequency(self, n=5000):
        # Addresses
        x = sorted(self.addr_list.keys(), key=lambda x: self.addr_list[x], reverse=True)[:n]
        y = [self.addr_list[X] for X in x]
        x = range(len(x))

        ax1 = plt.subplot(311)
        plt.plot(x, y)
        plt.ylim(0, y[0])
        plt.ylabel('Address Frequency')

        # Delta
        x = sorted(self.delta_list.keys(), key=lambda x: self.delta_list[x], reverse=True)[:n]
        y = [self.delta_list[X] for X in x]
        x = range(len(x))

        ax2 = plt.subplot(312, sharex=ax1)
        plt.plot(x, y)
        plt.ylim(0, y[0])
        plt.ylabel('Delta Frequency')

        # PC
        x = sorted(self.pc_list.keys(), key=lambda x: self.pc_list[x], reverse=True)[:n]
        y = [self.pc_list[X] for X in x]
        x = range(len(x))

        ax3 = plt.subplot(313, sharex=ax1)
        plt.plot(x, y)
        plt.xlim(0, 100)
        plt.ylim(0, y[0])
        plt.ylabel('PC Frequency')

        plt.show()
        
            
def print_summary(traces):
    traces[2].plot_frequency()
    sys.exit(1)
    print("-- Trace Summary --")
    print("Number of Load Instructions")
    for trace in traces:
        print(trace.benchmark + ": " + format(trace.num_loads, ","))
    print("")

    print("Num Unique Addresses:")
    for trace in traces:
        print(trace.benchmark + ": " + format(len(trace.addr_list.keys()), ","))
    print("")

    print("Num Unique Deltas:")
    for trace in traces:
        print(trace.benchmark + ": " + format(len(trace.delta_list.keys()), ","))
    print("")

    print("Num Unique PC:")
    for trace in traces:
        print(trace.benchmark + ": " + format(len(trace.pc_list.keys()),","))
    print("")

    print("Addresses Needed for 50% Coverage: ")
    for trace in traces:
        print(trace.benchmark + ": " + format(trace.needed_for_coverage(trace.addr_list),","))
    print("")

    print("Deltas Needed for 50% Coverage: ")
    for trace in traces:
        print(trace.benchmark + ": " + format(trace.needed_for_coverage(trace.delta_list),","))
    print("")

    print("Coverage from 50000 Addresses: ")
    for trace in traces: 
        print(trace.benchmark + ": " + str(round(trace.coverage_from_num(trace.addr_list, 50000) * 100, 3)) + "%")
    print("")

    print("Coverage from 50000 Deltas: ")
    for trace in traces: 
        print(trace.benchmark + ": " + str(round(trace.coverage_from_num(trace.delta_list, 50000) * 100, 3)) + "%")


def main(files):
    
    traces = []

    for f in files:
        trace = Trace()
        name = f[:f.index(".")]
        if os.path.exists(os.path.join(os.getcwd()+"/", name+".dump")):
            trace.load(name+".dump")
        else:
            trace.crawl(f)
            trace.save(name+".dump")
        traces.append(trace)

    print_summary(traces)

if __name__ == '__main__':
    main(sys.argv[1:])
