import cv2
import numpy as np
import serial
import time
import math

ser = serial.Serial('COM4',9600)
resx = []
resy = []
global array
array  = [[-1 for k in range(10)] for l in range(2)]
yt = []
ys = []
bs = []
resources = []
global xp
global yp
global xb
global yb
xp=0
yp = 0
xb = 0
yb=0
obs = []
cx=[1]
cy=[0]
px=[0]
py=[0]
mx = 0
my = 0
bsx = 0
bsy = 0
cxobs = []
cyobs = []
size  = 20
lowerr = [0,100,30]
upperr = [10,255,255]

#green
lowerg = [50,22,132]
upperg = [85,255,236]

#yellow
lowery = [20,30,30]
uppery = [40,255,255]

# brown
lower_b = [0,0,140]
upper_b = [39,35,255]

#blue
lower = [80,10,145]
upper = [120,255,255]

#pink
lp=[125,10,30]
up=[255,255,255]
cam = cv2.VideoCapture(1)
#cam.set(12, 4)
#cam.set(10, 0)
#cam.set(11, 44)

while True :
    
    ret, snap = cam.read()
    #snap = cv2.GaussianBlur(snap, (9,9),0)
    hsv = cv2.cvtColor(snap, cv2.COLOR_BGR2HSV)
    #yellow
    lower_y = np.array(lowery)
    upper_y = np.array(uppery)
    mask = cv2.inRange(hsv, lower_y, upper_y)
    res = cv2.bitwise_and(snap, snap, mask = mask)
    gray = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)

    ret, thresh = cv2.threshold(gray, 127, 255, 1)
    contours, h = cv2.findContours(thresh, 1, 2)

    for cnt in contours :
        if cv2.contourArea(cnt) > 100 and cv2.contourArea(cnt) < 100000 :
            approx = cv2.approxPolyDP(cnt, 0.1 * cv2.arcLength(cnt, True), True)
            if len(approx) == 3 :
                cv2.drawContours(snap,[cnt],-1,(25,90,0),2)
            if len(approx) == 4 :
                cv2.drawContours(snap,[cnt],-1,(25,90,0),2)

    lowerb = np.array(lower_b)
    upperb = np.array(upper_b)
    #brown
    mask = cv2.inRange(hsv, lowerb, upperb)
    res = cv2.bitwise_and(snap, snap, mask = mask)
    gray = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(gray, 127, 255, 1)
    contours1, h = cv2.findContours(thresh, 1, 2)
    for cnt in contours1 :
        if cv2.contourArea(cnt) > 500 and cv2.contourArea(cnt) < 10000 :
            approx = cv2.approxPolyDP(cnt, 0.1 * cv2.arcLength(cnt, True),True)
            if len(approx) == 4 :
                cv2.drawContours(snap,[cnt],-1,(255,255,255),2)
    #cv2.imshow('res', res)
    #cv2.imshow('mask', mask)
    
    #blue
    lowerob = np.array(lower)
    upperob = np.array(upper)
    mask = cv2.inRange(hsv, lowerob, upperob)
    res = cv2.bitwise_and(snap, snap, mask = mask)
    gray = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(gray, 127, 255, 1)
    contour2, h = cv2.findContours(thresh, 1, 2)
    for cnt in contour2:
        if cv2.contourArea(cnt) > 100 and cv2.contourArea(cnt) < 10000:
            approx = cv2.approxPolyDP(cnt, 0.1 * cv2.arcLength(cnt,True), True)
            if len(approx) == 4 :
                cv2.drawContours(snap, [cnt], -1,(70,0,0),2)
            
    cv2.imshow('snap', snap)
    k = cv2.waitKey(30) & 0xFF
    if k == 27:
        break
cv2.destroyAllWindows()



"""
countours - yellow
contours1 - brown
contours2 - blue"""
# mid point of resources
for cnt in contours:
    if cv2.contourArea(cnt) > 130 and cv2.contourArea(cnt) < 10000:
        approx = cv2.approxPolyDP(cnt, 0.1*cv2.arcLength(cnt,True), True)
        if len(approx) == 3:
            yt = yt + [cnt[0]]
            M=cv2.moments(cnt)
            x=int(M['m10']/M['m00'])
            y=int(M['m01']/M['m00'])
            resx = resx + [x/size]
            resy = resy + [y/size]
            px=px+[cnt[0]]
            cv2.drawContours(snap,[cnt],-1,(255,0,0),2)

        if len(approx) == 4:
            ys = ys + [cnt[0]]
            M=cv2.moments(cnt)
            x=int(M['m10']/M['m00'])
            y=int(M['m01']/M['m00'])
            resx = resx + [x/size]
            resy = resy + [y/size]
            cv2.drawContours(snap,[cnt],-1,(255,0,0),2)

resources = yt + ys


for cnt in contours1 :
    if cv2.contourArea(cnt) > 500 and cv2.contourArea(cnt) < 300000 :
        approx = cv2.approxPolyDP(cnt, 0.1 * cv2.arcLength(cnt, True),True)
        if len(approx) == 4 :
            M=cv2.moments(cnt)
            bsx = int(M['m10']/M['m00'])
            bsy = int(M['m01']/M['m00'])
for cnt in contour2:
    if cv2.contourArea(cnt) > 80 and cv2.contourArea(cnt) < 10000:
        approx = cv2.approxPolyDP(cnt, 0.1*cv2.arcLength(cnt, True), True)
        if len(approx) == 4:
            M=cv2.moments(cnt)
            x = int(M['m10']/M['m00'])
            y = int(M['m01']/M['m00'])
            cxobs = cxobs + [x/size]
            cyobs = cyobs + [y/size]
bsyc = bsx / size
bsxc = bsy / size
print "town hall : ",bsxc*size, bsyc*size
h = 0
w = 0
def marker():
    global xp
    global yp
    global xb
    global yb
    global size
    cx=[xp]
    cy=[yp]
    px=[xb]
    py=[yb]
    ret, frame = cam.read()
    cv2.imshow('initial', frame)
    frame = cv2.GaussianBlur(frame, (3,3),0)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_blue = np.array(lowerg)
    upper_blue = np.array(upperg)
    mask = cv2.inRange(hsv, lower_blue, upper_blue)
    res = cv2.bitwise_and(frame,frame, mask= mask)

    gray = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(gray,127,255,1)
    contours,h = cv2.findContours(thresh,1,2)
    for cnt in contours :
        if cv2.contourArea(cnt) > 70 and cv2.contourArea(cnt) < 4000 :
            approx = cv2.approxPolyDP(cnt,0.1*cv2.arcLength(cnt,True),True)
            if len(approx)== 4 :
                if len(approx)== 4:
                    M=cv2.moments(cnt)
                    x=int(M['m10']/M['m00'])
                    y=int(M['m01']/M['m00'])
                    cx = [x]
                    cy = [y]
                    cv2.drawContours(frame,[cnt],-1,(255,0,0),2)

    lowerp = np.array(lp)
    upperp = np.array(up)
    mask1 = cv2.inRange(hsv, lowerp, upperp)
    res1 = cv2.bitwise_and(frame,frame, mask= mask1)

    gray = cv2.cvtColor(res1, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(gray,127,255,1)
    contours,h = cv2.findContours(thresh,1,2)
    for cnt in contours :
        if cv2.contourArea(cnt) > 70 and cv2.contourArea(cnt) < 4000 :
            approx = cv2.approxPolyDP(cnt,0.1*cv2.arcLength(cnt,True),True)
            if len(approx)== 4 :
                M=cv2.moments(cnt)
                g=int(M['m10']/M['m00'])
                f=int(M['m01']/M['m00'])
                px = [g]
                py = [f]
                cv2.drawContours(frame,[cnt],-1,(255,255,0),2)
    i = 0
    j = 0
    w = 0
    h = 0
    for i in range(column):
        cv2.line(frame,(w,0),(w, height),(255,255,255),2)
        w+=size
    for j in range(row):
        cv2.line(frame,(0,h),(width,h),(255,255,255),2)
        h+=size
    for r in range(10) :
        cv2.rectangle(frame, (array[0][r],array[1][r]),(array[0][r]+2,array[1][r]+2),2,-1)
    for u in range(len(resources)):
        cv2.rectangle(frame, (resx[u]*size,resy[u]*size),(resx[u]*size+2,resy[u]*size+2),(0,255,0),2,-1)
    r = 0
    for r in range(len(cxobs)):
        cv2.rectangle(frame,(cxobs[r]*20,cyobs[r]*20),(cxobs[r]*20+2,cyobs[r]*20+2),(0,0,255),2,-1)
    cv2.imshow('frame', frame)
    xp=px[-1]
    
    yp=py[-1]
    
    xb=cx[-1]
    
    yb=cy[-1]
    #global mx
    #mx = int(cx[-1] + px[-1])/2
    #global my
    #my = int(cy[-1] + py[-1])/2

height, width, channel = snap.shape
def angle(xp,yp,xb,yb):
    dx=float(xp-xb)
    dy=float(yp-yb)
    if dx == 0:
        return 90
    global mtan
    if(dx > 0 and dy > 0):
        mtan=math.degrees(math.atan(float(dy/dx)))
    elif(dy>0 and dx<0):
        mtan=180 + math.degrees(math.atan(float(dy/dx)))
    elif(dy <0 and dx<0):
        mtan=180+math.degrees(math.atan(float(dy/dx)))
    else:
        mtan=360+math.degrees(math.atan(float(dy/dx)))

    return mtan
def distance(x1,y1,x2,y2) :
    dist = math.sqrt((x1- x2)**2 + (y1 - y2)**2)
    return dist
f = 0

row = height / size
column = width / size
grid = [[-1 for j in range(column)] for i in range(row)]
a1 = [bsxc]
a2 = [bsyc]
a3 = [0]
grid[bsxc][bsyc] = 0
obs  = [[0 for k in range(len(cxobs))] for l in range(2)]
r = 0
print len(cxobs)
for r in range(len(cxobs)) :
    obs[0][r] = cyobs[r]
    obs[1][r] = cxobs[r]
    print obs[0][r],obs[1][r]
i = 0

for i in range(len(cxobs)):
    xc = obs[0][i]
    yc = obs[1][i]
    grid[xc][yc] = -2
    if xc + 1 < row:
        grid[xc+1][yc] =  -2
        if yc + 1  <column:
            grid[xc + 1][yc +1] = -2
            grid[xc][yc +1] = -2
        if yc -1 >= 0:
            grid[xc +1][yc - 1] = -2
            grid[xc][yc - 1] = -2
    if xc - 1>=0:
        grid[xc-1][yc] =  -2
        if yc + 1  <column:
            grid[xc - 1][yc +1] = -2
            grid[xc][yc +1] = -2
        if yc -1 >= 0:
            grid[xc -1][yc - 1] = -2
            grid[xc][yc - 1] = -2
    for i in range(column-1):
        grid[0][i]=-2
        grid[row-1][i]=-2
    for j in range(row-1):
        grid[j][0]=-2
        grid[j][column-1]=-2
while len(a1) != 0:
    xc = a1.pop()
    yc = a2.pop()
    zc = a3.pop()
    if xc+ 1 < row and grid[xc+1][yc] == -1:
        a1 = [xc + 1] + a1
        a2 = [yc] + a2
        a3 = [zc + 1] + a3
        grid[xc + 1][yc] = zc + 1
    if xc - 1 >= 0 and grid[xc-1][yc] == -1:
        a1 = [xc - 1] + a1
        a2 = [yc] + a2
        a3 = [zc + 1] + a3
        grid[xc - 1][yc] = zc + 1
    if yc + 1 < column and grid[xc][yc+1] == -1:
        a1 = [xc] + a1
        a2 = [yc + 1] + a2
        a3 = [zc + 1] + a3
        grid[xc][yc + 1] = zc + 1
    if yc - 1 >= 0 and grid[xc][yc - 1] == -1:
        a1 = [xc] + a1
        a2 = [yc - 1] + a2
        a3 = [zc + 1] + a3
        grid[xc][yc - 1] = zc + 1
for h in range(len(resources)):
    xd = resy[h]
    yd = resx[h]
    grid[xd][yd] = 100
print ""
i = 0
for i in range(row) :
    print grid[i][:]
def arrayrev():
    global array
    rev= [[-1 for k in range(10)] for l in range(2)]
    r = 0
    while array[0][r] != -1:
        #rev = rev + [array.pop]
        r+=1
    t = 0
    r-=1
    while r >= 0:
        rev[0][t] = array[0][r]
        rev[1][t] = array[1][r]
        r-=1
        t+=1
    array = rev
    print array

def path(xc,yc):
    print "seed points : ",xc,yc
    global array
    array  = [[-1 for k in range(10)] for l in range(2)]
    r = 0
    array[1][r] = xc * size
    array[0][r] = yc * size

    r += 1
    d=grid[xc][yc]
    print d,xc,yc
    while d != 0:
        #down
        if yc + 1 < column and grid[xc][yc + 1] != -2 and grid[xc][yc +1] < grid[xc][yc]:
            while yc + 1 < column and grid[xc][yc + 1] != -2 and grid[xc][yc +1] < grid[xc][yc]:
                yc += 1
            d = grid[xc][yc]
            print d,xc,yc
            array[1][r] = xc * size
            array[0][r] = yc * size
            r += 1
        #right
        if xc + 1 < row and grid[xc+1][yc] != -2and grid[xc+1][yc] < grid[xc][yc]:
            while xc + 1 < row and grid[xc + 1][yc] != -2 and grid[xc+1][yc] < grid[xc][yc]:
                xc += 1
            d = grid[xc][yc]
            print d,xc,yc
            array[1][r] = xc * size
            array[0][r] = yc * size
            r += 1
        if yc - 1 >= 0 and grid[xc][yc -1] != -2 and grid[xc][yc -1] < grid[xc][yc]:
            while yc - 1 >=0 and grid[xc][yc - 1] != -2 and grid[xc][yc -1] < grid[xc][yc]:
                yc -= 1
            d = grid[xc][yc]
            print d,xc,yc
            array[1][r] = xc * size
            array[0][r] = yc * size
            r += 1
        if xc - 1 >= 0 and grid[xc - 1][yc] != -2 and grid[xc - 1][yc] < grid[xc][yc]:
            while xc - 1 >= 0 and grid[xc - 1][yc] != -2 and grid[xc - 1][yc] < grid[xc][yc]:
                xc -= 1
            d = grid[xc][yc]
            print d,xc,yc
            array[1][r] = xc * size
            array[0][r] = yc * size
            r += 1
        print array
r = 0


marker()
tol = 5
thresh = 40
slpt = 0.01
ro = 0
path(resy[ro], resx[ro])
arrayrev()
x1 = array[0][r]
y1 = array[1][r]
i = 0
'''for i in range(row) :
    print grid[i][:]'''
while True:
    marker()    
    mx=int(xp+xb)/2
    my=int(yp+yb)/2
    dd=distance(x1, y1, mx, my)
    t1 = angle(x1, y1,mx, my)
    t2 = angle(xp, yp, xb, yb)
    #print t1, t2
    diff=t1- t2

    if(diff <180):
        pass
    else:
        diff=0-(360-diff)
    if(diff > - 180):
        pass
    else:
        diff=360+diff
    #print "differnce", diff  
    if diff > tol :
        if diff > thresh:
            ser.write('l')
        else :
            ser.write('i')
        #print "l"
        time.sleep(slpt)
    elif diff < -tol :
        if diff  < - thresh :
            ser.write('r')
        else :
            ser.write('h')
        time.sleep(slpt)
        #print "r"
    else :
        ser.write('f')
        #print "f"
        time.sleep(slpt)
    
    if(dd<20):
        if True:
            ser.write('s')
            time.sleep(slpt)
            print "stop"
            
            if r < 9: 
                r += 1
            else:
                r = 0
            if array[0][r]!=-1:
                    
                x1 = array[0][r]
                y1 = array[1][r]
                print x1, y1
            else :
                if array[0][r - 1]==bsyc*size and array[1][r - 1]==bsxc*size:
                    ro = (ro + 1)%(len(resources) - 1)
                    path(resy[ro], resx[ro])
                    arrayrev()
                    ser.write('o')
                    r = 0
                else :
                    arrayrev()
                    ser.write('o')
                    r = 0

            print "blink"
            
    k = cv2.waitKey(30) & 0xFF
    if k == 27:
        ser.write('s')
        break
cam.release()
cv2.destroyAllWindows()
ser.close()