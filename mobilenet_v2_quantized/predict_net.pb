
mobilenet_v2_quant
data1 "	NCHW2NHWC8
13 "Int8Quantize*
Y_scalec��<*
Y_zero_pointr�
3
4
57 "Int8Conv*

kernel*

stride*
pad*
order"NHWC*
Y_zero_point *
Y_scale1;M=2 :NNPACK,DEPTHWISE_3x3G
79 "Int8Relu*
order"NHWC*
Y_scale1;M=*
Y_zero_point 2 : �
9
10
1113 "Int8Conv*

kernel*

stride*
pad *
order"NHWC*
Y_zero_point *
Y_scale1��=2 :NNPACK,DEPTHWISE_3x3I
1315 "Int8Relu*
order"NHWC*
Y_scale1��=*
Y_zero_point 2 : �
15
16
1719 "Int8Conv*

kernel*
pad*	
group *
order"NHWC*

stride*
Y_zero_point *
Y_scale ��=2 :NNPACK,DEPTHWISE_3x3I
1921 "Int8Relu*
order"NHWC*
Y_scale ��=*
Y_zero_point 2 : �
21
22
2325 "Int8Conv*

kernel*

stride*
pad *
order"NHWC*
Y_zero_point~*
Y_scaled�<2 :NNPACK,DEPTHWISE_3x3�
25
26
2729 "Int8Conv*

kernel*

stride*
pad *
order"NHWC*
Y_zero_point *
Y_scale��=2 :NNPACK,DEPTHWISE_3x3I
2931 "Int8Relu*
order"NHWC*
Y_scale��=*
Y_zero_point 2 : �
31
32
3335 "Int8Conv*

kernel*
pad*	
group`*
order"NHWC*

stride*
Y_zero_point *
Y_scale���=2 :NNPACK,DEPTHWISE_3x3I
3537 "Int8Relu*
order"NHWC*
Y_scale���=*
Y_zero_point 2 : �
37
38
3958 "Int8Conv*

kernel*

stride*
pad *
order"NHWC*
Y_zero_point�*
Y_scale�K�<2 :NNPACK,DEPTHWISE_3x3�
58
42
4345 "Int8Conv*

kernel*

stride*
pad *
order"NHWC*
Y_zero_point *
Y_scale}m�<2 :NNPACK,DEPTHWISE_3x3I
4547 "Int8Relu*
order"NHWC*
Y_scale}m�<*
Y_zero_point 2 : �
47
48
4951 "Int8Conv*

kernel*
pad*

group�*
order"NHWC*

stride*
Y_zero_point *
Y_scale�#]=2 :NNPACK,DEPTHWISE_3x3I
5153 "Int8Relu*
order"NHWC*
Y_scale�#]=*
Y_zero_point 2 : �
53
54
5557 "Int8Conv*

kernel*

stride*
pad *
order"NHWC*
Y_zero_point�*
Y_scaleGi�<2 :NNPACK,DEPTHWISE_3x3M
57
5860 "Int8Sum*
Y_scale*=*
Y_zero_point�*
order"NHWC2 : �
60
61
6264 "Int8Conv*

kernel*

stride*
pad *
order"NHWC*
Y_zero_point *
Y_scaley/=2 :NNPACK,DEPTHWISE_3x3I
6466 "Int8Relu*
order"NHWC*
Y_scaley/=*
Y_zero_point 2 : �
66
67
6870 "Int8Conv*

kernel*
pad*

group�*
order"NHWC*

stride*
Y_zero_point *
Y_scaleِC=2 :NNPACK,DEPTHWISE_3x3I
7072 "Int8Relu*
order"NHWC*
Y_scaleِC=*
Y_zero_point 2 : �
72
73
7493 "Int8Conv*

kernel*

stride*
pad *
order"NHWC*
Y_zero_point�*
Y_scaleDʜ<2 :NNPACK,DEPTHWISE_3x3�
93
77
7880 "Int8Conv*

kernel*

stride*
pad *
order"NHWC*
Y_zero_point *
Y_scale���<2 :NNPACK,DEPTHWISE_3x3I
8082 "Int8Relu*
order"NHWC*
Y_scale���<*
Y_zero_point 2 : �
82
83
8486 "Int8Conv*

kernel*
pad*

group�*
order"NHWC*

stride*
Y_zero_point *
Y_scale�f=2 :NNPACK,DEPTHWISE_3x3I
8688 "Int8Relu*
order"NHWC*
Y_scale�f=*
Y_zero_point 2 : �
88
89
9092 "Int8Conv*

kernel*

stride*
pad *
order"NHWC*
Y_zero_pointo*
Y_scaleD��<2 :NNPACK,DEPTHWISE_3x3N
92
93112 "Int8Sum*
Y_scale�c�<*
Y_zero_point�*
order"NHWC2 : �
112
96
9799 "Int8Conv*

kernel*

stride*
pad *
order"NHWC*
Y_zero_point *
Y_scaleD�<2 :NNPACK,DEPTHWISE_3x3J
99101 "Int8Relu*
order"NHWC*
Y_scaleD�<*
Y_zero_point 2 : �
101
102
103105 "Int8Conv*

kernel*
pad*

group�*
order"NHWC*

stride*
Y_zero_point *
Y_scaleq
�<2 :NNPACK,DEPTHWISE_3x3K
105107 "Int8Relu*
order"NHWC*
Y_scaleq
�<*
Y_zero_point 2 : �
107
108
109111 "Int8Conv*

kernel*

stride*
pad *
order"NHWC*
Y_zero_point{*
Y_scale\֍<2 :NNPACK,DEPTHWISE_3x3P
111
112114 "Int8Sum*
Y_scale_G�<*
Y_zero_point�*
order"NHWC2 : �
114
115
116118 "Int8Conv*

kernel*

stride*
pad *
order"NHWC*
Y_zero_point *
Y_scale��=2 :NNPACK,DEPTHWISE_3x3K
118120 "Int8Relu*
order"NHWC*
Y_scale��=*
Y_zero_point 2 : �
120
121
122124 "Int8Conv*

kernel*
pad*

group�*
order"NHWC*

stride*
Y_zero_point *
Y_scale�jF=2 :NNPACK,DEPTHWISE_3x3K
124126 "Int8Relu*
order"NHWC*
Y_scale�jF=*
Y_zero_point 2 : �
126
127
128147 "Int8Conv*

kernel*

stride*
pad *
order"NHWC*
Y_zero_point�*
Y_scale��<2 :NNPACK,DEPTHWISE_3x3�
147
131
132134 "Int8Conv*

kernel*

stride*
pad *
order"NHWC*
Y_zero_point *
Y_scale��<2 :NNPACK,DEPTHWISE_3x3K
134136 "Int8Relu*
order"NHWC*
Y_scale��<*
Y_zero_point 2 : �
136
137
138140 "Int8Conv*

kernel*
pad*

group�*
order"NHWC*

stride*
Y_zero_point *
Y_scale���<2 :NNPACK,DEPTHWISE_3x3K
140142 "Int8Relu*
order"NHWC*
Y_scale���<*
Y_zero_point 2 : �
142
143
144146 "Int8Conv*

kernel*

stride*
pad *
order"NHWC*
Y_zero_points*
Y_scalem��<2 :NNPACK,DEPTHWISE_3x3O
146
147166 "Int8Sum*
Y_scale�_�<*
Y_zero_pointz*
order"NHWC2 : �
166
150
151153 "Int8Conv*

kernel*

stride*
pad *
order"NHWC*
Y_zero_point *
Y_scaleW�<2 :NNPACK,DEPTHWISE_3x3K
153155 "Int8Relu*
order"NHWC*
Y_scaleW�<*
Y_zero_point 2 : �
155
156
157159 "Int8Conv*

kernel*
pad*

group�*
order"NHWC*

stride*
Y_zero_point *
Y_scale��=2 :NNPACK,DEPTHWISE_3x3K
159161 "Int8Relu*
order"NHWC*
Y_scale��=*
Y_zero_point 2 : �
161
162
163165 "Int8Conv*

kernel*

stride*
pad *
order"NHWC*
Y_zero_point�*
Y_scale�b <2 :NNPACK,DEPTHWISE_3x3O
165
166185 "Int8Sum*
Y_scale�0�<*
Y_zero_pointy*
order"NHWC2 : �
185
169
170172 "Int8Conv*

kernel*

stride*
pad *
order"NHWC*
Y_zero_point *
Y_scale��<2 :NNPACK,DEPTHWISE_3x3K
172174 "Int8Relu*
order"NHWC*
Y_scale��<*
Y_zero_point 2 : �
174
175
176178 "Int8Conv*

kernel*
pad*

group�*
order"NHWC*

stride*
Y_zero_point *
Y_scale�e3=2 :NNPACK,DEPTHWISE_3x3K
178180 "Int8Relu*
order"NHWC*
Y_scale�e3=*
Y_zero_point 2 : �
180
181
182184 "Int8Conv*

kernel*

stride*
pad *
order"NHWC*
Y_zero_point�*
Y_scale�f!=2 :NNPACK,DEPTHWISE_3x3P
184
185187 "Int8Sum*
Y_scale��.=*
Y_zero_point�*
order"NHWC2 : �
187
188
189191 "Int8Conv*

kernel*

stride*
pad *
order"NHWC*
Y_zero_point *
Y_scale�F�<2 :NNPACK,DEPTHWISE_3x3K
191193 "Int8Relu*
order"NHWC*
Y_scale�F�<*
Y_zero_point 2 : �
193
194
195197 "Int8Conv*

kernel*
pad*

group�*
order"NHWC*

stride*
Y_zero_point *
Y_scale,�-=2 :NNPACK,DEPTHWISE_3x3K
197199 "Int8Relu*
order"NHWC*
Y_scale,�-=*
Y_zero_point 2 : �
199
200
201220 "Int8Conv*

kernel*

stride*
pad *
order"NHWC*
Y_zero_pointy*
Y_scaled�j<2 :NNPACK,DEPTHWISE_3x3�
220
204
205207 "Int8Conv*

kernel*

stride*
pad *
order"NHWC*
Y_zero_point *
Y_scale�e�<2 :NNPACK,DEPTHWISE_3x3K
207209 "Int8Relu*
order"NHWC*
Y_scale�e�<*
Y_zero_point 2 : �
209
210
211213 "Int8Conv*

kernel*
pad*

group�*
order"NHWC*

stride*
Y_zero_point *
Y_scale�Z=2 :NNPACK,DEPTHWISE_3x3K
213215 "Int8Relu*
order"NHWC*
Y_scale�Z=*
Y_zero_point 2 : �
215
216
217219 "Int8Conv*

kernel*

stride*
pad *
order"NHWC*
Y_zero_point~*
Y_scale��<2 :NNPACK,DEPTHWISE_3x3P
219
220239 "Int8Sum*
Y_scale�+�<*
Y_zero_point�*
order"NHWC2 : �
239
223
224226 "Int8Conv*

kernel*

stride*
pad *
order"NHWC*
Y_zero_point *
Y_scale�Y=2 :NNPACK,DEPTHWISE_3x3K
226228 "Int8Relu*
order"NHWC*
Y_scale�Y=*
Y_zero_point 2 : �
228
229
230232 "Int8Conv*

kernel*
pad*

group�*
order"NHWC*

stride*
Y_zero_point *
Y_scale>=2 :NNPACK,DEPTHWISE_3x3K
232234 "Int8Relu*
order"NHWC*
Y_scale>=*
Y_zero_point 2 : �
234
235
236238 "Int8Conv*

kernel*

stride*
pad *
order"NHWC*
Y_zero_point*
Y_scale>��<2 :NNPACK,DEPTHWISE_3x3P
238
239241 "Int8Sum*
Y_scaleZ0(=*
Y_zero_point�*
order"NHWC2 : �
241
242
243245 "Int8Conv*

kernel*

stride*
pad *
order"NHWC*
Y_zero_point *
Y_scaleX�/=2 :NNPACK,DEPTHWISE_3x3K
245247 "Int8Relu*
order"NHWC*
Y_scaleX�/=*
Y_zero_point 2 : �
247
248
249251 "Int8Conv*

kernel*
pad*

group�*
order"NHWC*

stride*
Y_zero_point *
Y_scale�M=2 :NNPACK,DEPTHWISE_3x3K
251253 "Int8Relu*
order"NHWC*
Y_scale�M=*
Y_zero_point 2 : �
253
254
255274 "Int8Conv*

kernel*

stride*
pad *
order"NHWC*
Y_zero_point~*
Y_scalekx<2 :NNPACK,DEPTHWISE_3x3�
274
258
259261 "Int8Conv*

kernel*

stride*
pad *
order"NHWC*
Y_zero_point *
Y_scale6�<2 :NNPACK,DEPTHWISE_3x3K
261263 "Int8Relu*
order"NHWC*
Y_scale6�<*
Y_zero_point 2 : �
263
264
265267 "Int8Conv*

kernel*
pad*

group�*
order"NHWC*

stride*
Y_zero_point *
Y_scale^��<2 :NNPACK,DEPTHWISE_3x3K
267269 "Int8Relu*
order"NHWC*
Y_scale^��<*
Y_zero_point 2 : �
269
270
271273 "Int8Conv*

kernel*

stride*
pad *
order"NHWC*
Y_zero_point�*
Y_scale?�<2 :NNPACK,DEPTHWISE_3x3O
273
274293 "Int8Sum*
Y_scale���<*
Y_zero_point{*
order"NHWC2 : �
293
277
278280 "Int8Conv*

kernel*

stride*
pad *
order"NHWC*
Y_zero_point *
Y_scale�o=2 :NNPACK,DEPTHWISE_3x3K
280282 "Int8Relu*
order"NHWC*
Y_scale�o=*
Y_zero_point 2 : �
282
283
284286 "Int8Conv*

kernel*
pad*

group�*
order"NHWC*

stride*
Y_zero_point *
Y_scale�5�<2 :NNPACK,DEPTHWISE_3x3K
286288 "Int8Relu*
order"NHWC*
Y_scale�5�<*
Y_zero_point 2 : �
288
289
290292 "Int8Conv*

kernel*

stride*
pad *
order"NHWC*
Y_zero_point�*
Y_scale�b�<2 :NNPACK,DEPTHWISE_3x3P
292
293295 "Int8Sum*
Y_scale���<*
Y_zero_point�*
order"NHWC2 : �
295
296
297299 "Int8Conv*

kernel*

stride*
pad *
order"NHWC*
Y_zero_point *
Y_scaleW�\=2 :NNPACK,DEPTHWISE_3x3K
299301 "Int8Relu*
order"NHWC*
Y_scaleW�\=*
Y_zero_point 2 : �
301
302
303305 "Int8Conv*

kernel*
pad*

group�*
order"NHWC*

stride*
Y_zero_point *
Y_scale��/=2 :NNPACK,DEPTHWISE_3x3K
305307 "Int8Relu*
order"NHWC*
Y_scale��/=*
Y_zero_point 2 : �
307
308
309311 "Int8Conv*

kernel*

stride*
pad *
order"NHWC*
Y_zero_point�*
Y_scale9vF<2 :NNPACK,DEPTHWISE_3x3�
311
312
313315 "Int8Conv*

kernel*

stride*
pad *
order"NHWC*
Y_zero_point *
Y_scale��)>2 :NNPACK,DEPTHWISE_3x3K
315317 "Int8Relu*
order"NHWC*
Y_scale��)>*
Y_zero_point 2 : j
317319 "Int8AveragePool*
order"NHWC*

kernel*

stride*
Y_scale��)>*
Y_zero_point 2 : S
319
320
321323 "Int8FC*
order"NHWC*
Y_zero_pointN*
Y_scaleh�B>2 : N
323325 "Int8Softmax*
order"NHWC*
Y_scale  �;*
Y_zero_point 2 :  
325softmax "Int8Dequantize:data:4:5:10:11:16:17:22:23:26:27:32:33:38:39:42:43:48:49:54:55:61:62:67:68:73:74:77:78:83:84:89:90:96:97:102:103:108:109:115:116:121:122:127:128:131:132:137:138:143:144:150:151:156:157:162:163:169:170:175:176:181:182:188:189:194:195:200:201:204:205:210:211:216:217:223:224:229:230:235:236:242:243:248:249:254:255:258:259:264:265:270:271:277:278:283:284:289:290:296:297:302:303:308:309:312:313:320:321Bsoftmax