# Инструкция для запуска
Для запуска необходимо переместить веса в папку <i>weights</i>, также в папке <i>data</i> (для примера) могут быть данные и необходимые файлы для работы. Примерная структура может быть такой:
```
    - data/
        - images/
        - jsons/
        - images_jsons.csv
        - orig_palettes.pickle
        - reference.yaml
    - weights/
    - scripts/
    - config.json
    - main.py
```
```
reference.yaml - файл с формой знака по классам, для работы детектора нарушения целостности
images_jsons.csv - файл с именем картинок и их json файлами (можно сделать свой по образу и подобию)
orig_palettes.pickle - файл с оригинальными палитрами, для детектора выцветания
```
Необходимые для работы пути можно заменить в файле <i>config.json</i>

## Установка (Docker)
```
git clone https://gitlab.nanosemantics.ru/na/cv/digital-roads/externaldigitalroads.git
cd externaldigitalroads/
git checkout defectdetector
docker build -t defectdetector .
```


Справка
```
./launch.sh -h 
```
Запуск
```
./launch.sh \
-i path/to/images -j path/to/jsons -w path/to/weights \
-mt <PyTorch / TensorRT>
```
После выполнения появится папка <i>output</i> c результатами 

## Установка
```
git clone https://gitlab.nanosemantics.ru/na/cv/digital-roads/externaldigitalroads.git
cd externaldigitalroads/
git checkout defectdetector
```

Установка зависимостей
```
pip install -r requirements.txt
```

Запуск скрипта
```
python main.py -ii <path to imgs folder> -ij <path to jsons folder> -icsv <path to input csv file> -o <path to output folder>
```

После выполнения в папке <i>output</i> появятся файлы в формате json, а также <i>output.csv</i> - файл с результатами по всем файлам 

#### Формат выходного json-а

| Поле               | Значение |
|--------------------|---|
| ‘image_name’       | Имя картинки.                 |
| 'discolor_defect'  | True (дефект есть)/False - наличие выцветания знака. Примечание: None - появляется если для данного типа знака нет эталлонной палитры.|
| ‘graffiti_defect’  | True (дефект есть)/False - наличие граффити на знаке.|
| ‘surface_defect’   | True (дефект есть)/False - наличие изменений лицевой поверхности знака. Примечание: None - появляется если знак круглый (в разработке).|





