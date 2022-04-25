
# -*- coding: utf-8 -*-

from flask import Flask, jsonify, request
from os import environ
from flask_cors import CORS
import numpy as np
import math
import mahotas as mt
from cv2 import cv2
import collections
import urllib.request
import pickle
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
app = Flask(__name__)

app.config['CORS_HEADERS'] = 'Content-Type'
app.config['CORS_RESOURCES'] = {r"/api/*": {"origins": "*"}}

cors = CORS(app)

cors = CORS(app, resources = {
    r"/*":{
        "origins":"*"
    }
})

# load
with open('model.pkl', 'rb') as f:
    clf1 = pickle.load(f)
    
# load
with open('normalizacion.pkl', 'rb') as f:
    scaler = pickle.load(f)    
    
 
def segmentar(imagen):
    ret,cierre = cv2.threshold(imagen,5,255,cv2.THRESH_BINARY_INV)
    # Copy the thresholded image.
    im_floodfill = cierre.copy()
    h, w = cierre.shape[:2]
    mask = np.zeros((h+2, w+2), np.uint8)
    # Floodfill from point (0, 0)
    fill = cv2.floodFill(im_floodfill, mask, (0,0), 255);
    # Invert floodfilled image
    im_floodfill_inv = cv2.bitwise_not(im_floodfill)
    # Combine the two images to get the foreground.
    im_out = cierre | im_floodfill_inv
    mgray = cv2.cvtColor(im_out, cv2.COLOR_BGR2GRAY)
    ret, binary = cv2.threshold(mgray, 250, 255, cv2.THRESH_BINARY)
    return binary
    

def extract_features(image):
    # calculate haralick texture features for 4 types of adjacency
    textures = mt.features.haralick(image)
    
    #print(textures)
    # take the mean of it and return it
    ht_mean = textures.mean(axis=0)
    #print(ht_mean)
    
    return ht_mean

def threshold_hsv(img, list_min_v, list_max_v, reverse_hue=False, use_s_prime=False):
    """
    Take BGR image (OpenCV imread result) and return thresholded image
    according to values on HSV (Hue, Saturation, Value)
    Pixel will worth 1 if a pixel has a value between min_v and max_v for all channels
    :param img: image BGR if rgb_space = False
    :param list_min_v: list corresponding to [min_value_H,min_value_S,min_value_V]
    :param list_max_v: list corresponding to [max_value_H,max_value_S,max_value_V]
    :param use_s_prime: Bool -> True if you want to use S channel as S' = S x V else classic
    :param reverse_hue: Useful for Red color cause it is at both extremum
    :return: threshold image
    """
    frame_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    if use_s_prime:
        frame_hsv[:, :, 1] = (1. / 255) * frame_hsv[:, :, 1] * frame_hsv[:, :, 2].astype(np.uint8)

    if not reverse_hue:
        return cv2.inRange(frame_hsv, tuple(list_min_v), tuple(list_max_v))
    else:
        list_min_v_c = list(list_min_v)
        list_max_v_c = list(list_max_v)
        lower_bound_red, higher_bound_red = sorted([list_min_v_c[0], list_max_v_c[0]])
        list_min_v_c[0], list_max_v_c[0] = 0, lower_bound_red
        low_red_im = cv2.inRange(frame_hsv, tuple(list_min_v_c), tuple(list_max_v_c))
        list_min_v_c[0], list_max_v_c[0] = higher_bound_red, 179
        high_red_im = cv2.inRange(frame_hsv, tuple(list_min_v_c), tuple(list_max_v_c))
        return cv2.addWeighted(low_red_im, 1.0, high_red_im, 1.0, 0) 

def contarColoresHSV(imagen, mascara):

    imgHSV = cv2.cvtColor(imagen, cv2.COLOR_BGR2HSV)
        
    #cv2.imwrite("HSV"+clase + nombreImagen +".jpg", imgHSV)

  
    rojo = threshold_hsv(imagen, (175, 70,165.75), (5, 255, 255), reverse_hue=True, use_s_prime=False)
    white = threshold_hsv(imagen,(0, 0, 229), (255, 10, 255),  reverse_hue=False, use_s_prime=False)
    
    black = threshold_hsv(imagen, (0, 0, 1), (255, 255, 40), reverse_hue=False, use_s_prime=True)
    light_brown = threshold_hsv(imagen, (0, 2, 140), (15, 255, 165.75), reverse_hue=False, use_s_prime=False)
    
    dark_brown = threshold_hsv(imagen, (0, 5, 40), (180, 255, 140), reverse_hue=False, use_s_prime=False)
    
    blue = threshold_hsv(imagen, (100, 20, 153), (110, 255, 255), reverse_hue=False, use_s_prime=False)
    
    """cv2.imwrite(clase + nombreImagen +"_rojo.jpg",rojo)
    cv2.imwrite(clase + nombreImagen +"_light_brown.jpg",light_brown)
    cv2.imwrite(clase +nombreImagen +"_dark_brown.jpg",dark_brown)
    cv2.imwrite(clase +nombreImagen +"_blue.jpg",blue)
    cv2.imwrite(clase +nombreImagen +"_white.jpg",white) 
    cv2.imwrite(clase +nombreImagen +"_black.jpg",black) 
    cv2.imwrite(clase +nombreImagen +"_aa_imagen.jpg",imagen) 
    """
    #final = np.hstack(( rojo, light_brown, dark_brown, blue, white, black))
    
    #cv2.imwrite(clase +nombreImagen +"_ab_final.jpg",  final)

    area_rojo = cv2.countNonZero(rojo)
    area_light_brown = cv2.countNonZero(light_brown)
    area_dark_brown = cv2.countNonZero(dark_brown)
    area_blue = cv2.countNonZero(blue)
    area_white = cv2.countNonZero(white)
    area_black = cv2.countNonZero(black)
    area_mask = cv2.countNonZero(mascara)

    p_white = (area_white/area_mask)*100
    p_red = (area_rojo/area_mask)*100
    p_light_brown = (area_light_brown/area_mask)*100
    p_dark_brown = (area_dark_brown/area_mask)*100
    p_blue = (area_blue/area_mask)*100
    p_black = (area_black/area_mask)*100
    
    count = 0
    
    if p_white >= 0.05:
        count = count + 1
    if p_red >= 0.05:
        count = count + 1    
    if p_light_brown >= 0.05:
        count = count + 1
    if p_dark_brown >= 0.05:
        count = count + 1
    if p_blue >= 0.05:
        count = count + 1
    if p_black >= 0.05:
        count = count + 1
    
    #porcentajes = [p_white,  p_red , p_light_brown, p_dark_brown, p_blue, p_black]
    
    print("\count\n")
    print(count)
    print("\n")
    
    return count, p_white, p_red, p_light_brown, p_dark_brown, p_blue, p_black


def white_balance(img):
    result = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    avg_a = np.average(result[:, :, 1])
    avg_b = np.average(result[:, :, 2])
    result[:, :, 1] = result[:, :, 1] - ((avg_a - 128) * (result[:, :, 0] / 255.0) * 1.1)
    result[:, :, 2] = result[:, :, 2] - ((avg_b - 128) * (result[:, :, 0] / 255.0) * 1.1)
    result = cv2.cvtColor(result, cv2.COLOR_LAB2BGR)
    return result

def DistanciaPuntos(coordinate1, coordinate2):
    return pow(pow(coordinate1[0] - coordinate2[0], 2) + pow(coordinate1[1] - coordinate2[1], 2), .5)

def AreaLesion(mascara):
    mascara=np.double(mascara)/255 #Hacer los valores a 1
    Ap=np.sum(mascara)
    return Ap

def CaracBordes(mascara):
    
    
    borde=cv2.Canny(mascara,100,200)
    
    kernel = np.ones((3,3),np.uint8)
    erosion = cv2.dilate(borde,kernel,iterations = 1)
    contours, hierarchy = cv2.findContours(erosion,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    
    count = 0
    cX0 = 0
    cY0 = 0
    for c in contours:
        # calculate moments for each contour
        M = cv2.moments(c)
        # calculate x,y coordinate of center
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
        
        if count == 0:
            cX0 = cX
            cY0 = cY
            print("cX %d cY %d" %( cX0, cY0 ))
 
        count = count + 1
        #SE OCUPA PARA DIBUJAR EL CENTROIDE
        centro=cv2.circle(erosion, (cX, cY), 1, (255, 255, 255), -1)
        print(type(cX))
        tagCentro = str(count) + "_" +str(cX) + "_" + str(cY) + "centro.png"
        #cv2.imwrite( tagCentro,centro)
  
    Distancia2 = np.where(borde > 0)
    
    [m,n]=borde.shape
    
    Distancia= []
    
    for x in range(0,m):
        for y in range(0,n):
            if borde[x][y] > 0:
                Distancia.append(DistanciaPuntos([cY0,cX0],[x,y]))
    
    
    #new_dict_comp = {(k[0],k[1]):DistanciaPuntos([cY0,cX0],[k[0],k[1]]) for (k,v) in borde if v == 1}
                
    #takewhile(lambda x: x>0, borde) 

    DistanciaMedia = np.sum(Distancia)/len(Distancia)
    
    #Diametro
    cnt = contours[0]
    distances = []
    distances1 = []
    Diametro= 0
    
    for i in range(len(cnt)-1):
        for j in range(i+1, len(cnt)):
            aux = DistanciaPuntos(cnt[i][0],cnt[j][0])
            if aux > Diametro:
                Diametro = aux
                distances = cnt[i][0]
                distances1 = cnt[j][0]
                
    #SE OCUPA PARA DIBUJAR EL CIRCULO QUE ENCIERRA A LA LESION 
    #h,w,t = mascara.shape
    mask_circulo = np.zeros((mascara.shape),np.uint8)
    #img2 =cv2.circle(erosion, (cX, cY), int(RadioMayor), (255, 255, 255), 1)
    img3 =cv2.circle(mask_circulo, (cX, cY), int(DistanciaMedia), (255, 255, 255), -1)

    area_externa = mascara - img3

    area_sobrante = img3 - mascara
    aso = AreaLesion(area_sobrante)

    area_interna = mascara - area_externa 
    ai = AreaLesion(area_interna)
    
    Asimetria = aso / ai
    
    Varianza=[]
    for x in range(0,len(Distancia)):
       Varianza.append( pow((Distancia[x]-DistanciaMedia),2))
    
    VarianzaMedia= np.sum(Varianza)/len(Varianza)
    DesviacionTipica= pow(VarianzaMedia,.5)

    return DesviacionTipica,Asimetria,Diametro
    
        
def AreaLesion2(mascara):
    mascara=np.double(mascara)/255 #Hacer los valores a 1
    Ap=np.sum(mascara)
    return Ap

def PerLesion(mascara):
    #im2, contours, hierarchy = cv2.findContours(mascara1,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    contours, hierarchy = cv2.findContours(mascara,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    cnt = contours[0]
    Pp = cv2.arcLength(cnt,True)
    return Pp

def Diametro(mascara):
    A= AreaLesion2(mascara)
    D_ima= math.sqrt((4 * A)/math.pi)
    D= (D_ima/20)* 0.2645

    return D

def CaracBordes2(mascara):
    ################################# Bordes Segundo #####################################
    A=AreaLesion2(mascara)
    P=PerLesion(mascara)
    B= (4 * math.pi * A) / (P**2)

    return B

def fracAsim(mascara):
    a=fraccionar(mascara,4)
    b=[]
    for i in a:
        b.append(mascara*i)
    return b

def fraccionar(mascara,n): 
    vx,vy,x,y=orientacion(mascara)
    maskFrac=fracAngulo(mascara,vx,vy,x,y,n)
    return maskFrac

def orientacion(mascara):
    mascara1=mascara.copy()
    contours, hierarchy = cv2.findContours(mascara1,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    cnt = contours[0]
    [vx,vy,x,y] = cv2.fitLine(cnt, cv2.DIST_L2,0,0.01,0.01)
    return vx,vy,x,y

def fracAngulo(mascara,vx,vy,x,y,n):
    rows,cols = mascara.shape[:2]
    a1=np.zeros((rows,cols))
    a2=np.zeros((rows,cols))
    k=90*(4/n)
    t=k*(np.pi/180)
    for i in range(0,rows):
        for j in range(0,cols):
            a1[i,j]=np.dot([i,j],[-vx[0],vy[0]])-np.dot([y[0],x[0]],[-vx[0],vy[0]])>0      
    [vx1,vy1]=[vx*np.cos(t)-vy*np.sin(t),vx*np.sin(t)+vy*np.cos(t)]
    if np.dot([vx1[0],vy1[0]],[vx[0],vy[0]])<0:
        [vx1[0],vy1[0]]=[-vx1[0],-vy1[0]]
    for i in range(0,rows):
        for j in range(0,cols):
            a2[i,j]=np.dot([i,j],[-vx1[0],vy1[0]])-np.dot([y[0],x[0]],[-vx1[0],vy1[0]])>0 
    vx=vx1
    vy=vy1
    af1=(a1-a2)>0
    fracciones=[af1]
    for i in range(1,n):
        M = cv2.getRotationMatrix2D((round(x[0]),round(y[0])),k*i,1)
        dst = cv2.warpAffine(np.uint8(af1),M,(cols,rows))
        fracciones.append(dst)
    return fracciones

def Simetria(mascara,angle):
    fracciones= fracAsim(mascara)
    
    height, width = mascara.shape
    wi=(width/2)
    he=(height/2)
    
    M = cv2.getRotationMatrix2D((wi, he), angle, 1.0)
    rotada0= cv2.warpAffine(fracciones[0], M, (width, height))
    rotada1= cv2.warpAffine(fracciones[1], M, (width, height))
    rotada2= cv2.warpAffine(fracciones[2], M, (width, height))
    rotada3= cv2.warpAffine(fracciones[3], M, (width, height))
    
    ParteSuperior= rotada0 + rotada3
    ParteInferior= rotada1 + rotada2
    ParteIzquierda= rotada0 + rotada1
    ParteDerecha= rotada2 + rotada3
    
    ParteSuperiorflipped= cv2.flip(ParteSuperior, 0) 
    ParteInferiorflipped= cv2.flip(ParteInferior, 0)
    ParteIzquierdaflipped= cv2.flip(ParteIzquierda, 1)
    ParteDerechaflipped= cv2.flip(ParteDerecha, 1)
    
    TX1= ParteSuperiorflipped + ParteSuperior
    TX2= ParteInferiorflipped + ParteInferior
    
    TY1= ParteDerecha + ParteDerechaflipped
    TY2= ParteIzquierda + ParteIzquierdaflipped
    
    X1= TX1 - mascara
    X2= TX2 - mascara
    
    Y1= TY1 - mascara
    Y2= TY2 - mascara
    
    AS1= AreaLesion(X1+X2) / AreaLesion(mascara)
    print(AS1)
    AS2=AreaLesion(Y1+Y2) / AreaLesion(mascara)
    print(AS2)    
    return AS1,AS2

def centrarImagen(image):  
    
    ##############################AGREGAR PIXELES A LAS IMAGENES#########################
    Copia= np.zeros((100,600))
    image= np.concatenate((image, Copia), axis=0)
    image= np.transpose(image)
    Copia1= np.zeros((100,550))
    image= np.concatenate((Copia1,image), axis=0)
    image= np.transpose(image)
    image= np.uint8(image)
    
    height, width = image.shape
    wi=(width/2)
    he=(height/2)
      
    M = cv2.moments(image)
    cX = int(M["m10"] / M["m00"])
    cY = int(M["m01"] / M["m00"])
      
    offsetX = (wi-cX)
    offsetY = (he-cY)
    T = np.float32([[1, 0, offsetX], [0, 1, offsetY]]) 
    centered_image = cv2.warpAffine(image, T, (width, height))
    
    ##################################### ROTAR IMAGEN ##################################
    
    #ENCONTRAR LOS CONTORNOS
    borde=cv2.Canny(centered_image,100,200)
    kernel = np.ones((3,3),np.uint8)
    dilate = cv2.dilate(borde,kernel,iterations = 1)
    contours, hierarchy = cv2.findContours(dilate,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    cnt = contours[0]
    
    ###### LINEA ##########
    rows,cols = image.shape[:2]
    Copia= np.zeros((rows,cols))
    centered =centered_image
    [vx,vy,x,y] = cv2.fitLine(cnt, cv2.DIST_L2,0,0.01,0.01)
    lefty = int((-x*vy/vx) + y)
    righty = int(((cols-x)*vy/vx)+y)
    img = cv2.line(Copia,(cols-1,righty),(0,lefty),(255,255,255),1)  
    img = np.uint8((img * centered) * 255)
    
    #############################################
    borde=cv2.Canny(img,100,200) 
    kernel = np.ones((3,3),np.uint8)
    dilate = cv2.dilate(borde,kernel,iterations = 1)
    contours, hierarchy = cv2.findContours(dilate,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    cnt = contours[0]
    
    #DETERMINAR EL ANGULO
    angle = cv2.minAreaRect(cnt)[-1]
    
    #AJUSTAR EL ANGULO
    if angle < -45:
        angle = (90 + angle)
    else:
       angle = angle
      
    return centered_image,angle


def seleccionarComponente(thresh):
    
    ret, labels = cv2.connectedComponents(thresh)

    Frecuencia = labels.flatten()
    Contador = collections.Counter(Frecuencia)
    print(Contador)
    
    max1=0
    max=0
    i = 1
    if len(Contador) > 1:
        while i < len(Contador):
          if max1 < Contador[i]:
              max1= Contador[i]
              
              max= i
          i += 1
    else:
        max=1
    
    lesion = (labels == max)
    lesion = np.uint8((lesion/max) * 255)
    return lesion  
    

def procesarImagen(imageColor, mascara):
    
    #imgray = cv2.cvtColor(mascara, cv2.COLOR_BGR2GRAY)
    #ret, thresh = cv2.threshold(imgray, 127, 255, 0)
    
    thresh = seleccionarComponente(mascara)
    
    imagenSegmentada = cv2.bitwise_and(imageColor,imageColor,mask = thresh)

    mask = thresh

    count, p_white, p_red, p_light_brown, p_dark_brown, p_blue, p_black = contarColoresHSV(imagenSegmentada, thresh)
    
    DE,A,D = CaracBordes(thresh)
    
    
    #imRecortada = recortarImagen(index)
    imagenGrisSegmentada = cv2.cvtColor(imagenSegmentada, cv2.COLOR_BGR2GRAY)
    feature = extract_features(imagenGrisSegmentada)

    rotada,angle=centrarImagen(thresh)  ########### ROTAR IMAGEN ###########

    Borde2 = CaracBordes2(thresh)  #NO ES NECESARIA LA IMAGEN ROTADA
    Diametro2= Diametro(thresh)   #NO ES NECESARIA LA IMAGEN ROTADA
    
    AS1,AS2= Simetria(rotada,angle) #SI ES NECESARIA LA IMAGEN ROTADA
    
    #Caracteristica_imagen = [DE,A,D, count, p_white, p_red, p_light_brown, p_dark_brown, p_blue, p_black, AS1,AS2,Borde2, Diametro2, feature[0],feature[1],feature[2],feature[3], feature[4], feature[5],feature[6],feature[7],feature[8],feature[9],feature[10],feature[11],feature[12]]
   

    return DE,A,D, count, p_white, p_red, p_light_brown, p_dark_brown, p_blue, p_black, AS1,AS2,Borde2, Diametro2, feature[0],feature[1],feature[2],feature[3], feature[4], feature[5],feature[6],feature[7],feature[8],feature[9],feature[10],feature[11],feature[12]




@app.route('/procesar_imagen', methods = ['POST', 'GET'])
def get_imagen_procesada():

    urllib.request.urlretrieve(
  'https://firebasestorage.googleapis.com/v0/b/reconocimiento-de-patron-d69cc.appspot.com/o/images%2Fmascara.png?alt=media&token=0591c4e7-d590-46e7-9a19-a69407bdd1fb',
   "imagen.png")
    
    img = cv2.imread("imagen.png")   
    segmentado = segmentar(img)
    cv2.imwrite('segmentada.png',segmentado)
    
    # load
    # with open('model.pkl', 'rb') as f:
    #     clf2 = pickle.load(f)
    
    
    clase = "prueba"
    
    DE,A,D, count, p_white, p_red, p_light_brown, p_dark_brown, p_blue, p_black, AS1,AS2,Borde2, Diametro2, T0,T1,T2,T3, T4, T5,T6,T7,T8,T9,T10,T11,T12 = procesarImagen( img , segmentado )
    
    
    X = [DE,A,D, count, p_white, p_red, p_light_brown, p_dark_brown, p_blue, p_black, AS1,AS2,Borde2, Diametro2, T0,T1,T2,T3, T4, T5,T6,T7,T8,T9,T10,T11,T12]
    
    a = np.array(X)
    col_vec = a.reshape(1, -1)
    # print( col_vec )

    col_vec = scaler.transform(col_vec)
    col_vec = pd.DataFrame(col_vec)
    
    pred2=clf1.predict(col_vec)
    
    #print( col_vec.iloc[0,:] )
    #print("\n")
    #print( col_vec.values )
    DE,A,D, count, p_white, p_red, p_light_brown, p_dark_brown, p_blue, p_black, AS1,AS2,Borde2, Diametro2, T0,T1,T2,T3, T4, T5,T6,T7,T8,T9,T10,T11,T12 = col_vec.iloc[0,:]
    
    if pred2 == 1 :
        clase = "Melanoma"
    if pred2 == 2 :
        clase = "Nevus melanocitico" 
    
       
       #DesviacionTipica,Asimetria,Diametro
    response = jsonify({'BORDE1': DE,
                        'ASIMETRIA1': A,
                        'DIAMETRO1': D,
                        'Conteo': count,
                        'Blanco': p_white,
                        'Rojo': p_red,
                        'CafeClaro': p_light_brown,
                        'CafeOscuro': p_dark_brown,
                        'Azul': p_blue,
                        'Negro': p_black,
                        'Asimetria2A': AS1,
                        'Asimetria2B': AS2,
                        'Borde2': Borde2,
                        'Diametro2': Diametro2,
                        'SegundoMomentAngular': T0,
                        'CONTRASTE': T1,
                        'CORRELACION': T2,
                        'SUM_CUAD': T3,
                        'MOMENT_DIF_INV': T4,
                        'SUM_PROM': T5,
                        'SUM_VAR': T6,
                        'SUM_ENTROP': T7,
                        'ENTROPIA': T8,
                        'DIF_VAR': T9,
                        'DIF_ENTROP': T10,
                        'MEDIDA_CORRELACION': T11,
                        'INF_CORRELACION': T12,
                        'Clase': clase,
                        
    })

    response.headers.add("Access-Control-Allow-Origin", "*")
    response.headers.add("Access-Control-Allow-Credentials", "true")
    return response, 200



if __name__ == "__main__":
    
    ENVIRONMENT_DEBUG = environ.get("APP_DEBUG", True)
    ENVIRONMENT_PORT = environ.get("APP_PORT", 4000)
    app.run(host='127.0.0.1', port=ENVIRONMENT_PORT, debug=ENVIRONMENT_DEBUG)
