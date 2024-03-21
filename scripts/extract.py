import sys
import re

if len(sys.argv) != 2:
    print("Usage: python3 " + sys.argv[0] + " [input file]")
    sys.exit()

file = open(sys.argv[1], 'r')
lines = file.readlines()
even = False
for line in lines:
    if even:
        print(line,end='')
        even = False
    if line.startswith('>>>'):
        experiment = re.sub('.*_guessing/','',line)
        dataset = re.sub('-.*\n','',experiment)
        experiment = re.sub('.*-i','',experiment)
        i = re.sub('u.*\n','',experiment)
        experiment = re.sub('.*u','',experiment)
        u = re.sub('g.*\n','',experiment)
        experiment = re.sub('.*g','',experiment)
        g = re.sub('_.*\n','',experiment)
        print("{},{},{},{},".format(dataset,i,u,g),end='')
        even = True
