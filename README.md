# Kultasieni deep learning model
![image of boletus edulis](https://github.com/timonenv/kultasieni/blob/main/images/image1.jpg)

Binary mushroom classifier (poisonous/edible).
Just for fun - always identify any mushroom 100% yourself before eating!

### Website / API 
API hosted at: [timonenv.github.io/kultasieni/](timonenv.github.io/kultasieni/)

### Mushroom Data Features

Data from: [https://www.kaggle.com/datasets/dhinaharp/mushroom-dataset/data](https://www.kaggle.com/datasets/dhinaharp/mushroom-dataset/data)
This table outlines the features used for the model and what can be input for prediction (n: nominal, m: metrical).

| Feature | Values |
| --- | --- |
| cap-diameter (m) | float number in cm |
| cap-shape (n) | bell=b, conical=c, convex=x, flat=f, sunken=s, spherical=p, others=o |
| cap-surface (n) | fibrous=i, grooves=g, scaly=y, smooth=s, shiny=h, leathery=l, silky=k, sticky=t, wrinkled=w, fleshy=e |
| cap-color (n) | brown=n, buff=b, gray=g, green=r, pink=p, purple=u, red=e, white=w, yellow=y, blue=l, orange=o, black=k |
| does-bruise-bleed (n) | bruises-or-bleeding=t, no=f |
| gill-attachment (n) | adnate=a, adnexed=x, decurrent=d, free=e, sinuate=s, pores=p, none=f, unknown=? |
| gill-spacing (n) | close=c, distant=d, none=f |
| gill-color (n) | see cap-color + none=f |
| stem-height (m) | float number in cm |
| stem-width (m) | float number in mm |
| stem-root (n) | bulbous=b, swollen=s, club=c, cup=u, equal=e, rhizomorphs=z, rooted=r |
| stem-surface (n) | see cap-surface + none=f |
| stem-color (n) | see cap-color + none=f |
| veil-type (n) | partial=p, universal=u |
| veil-color (n) | see cap-color + none=f |
| has-ring (n) | ring=t, none=f |
| ring-type (n) | cobwebby=c, evanescent=e, flaring=r, grooved=g, large=l, pendant=p, sheathing=s, zone=z, scaly=y, movable=m, none=f, unknown=? |
| spore-print-color (n) | see cap-color |
| habitat (n) | grasses=g, leaves=l, meadows=m, paths=p, heaths=h, urban=u, waste=w, woods=d |
| season (n) | spring=s, summer=u, autumn=a, winter=w |
