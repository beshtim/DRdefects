# Инструкция для запуска
Для запуска необходимо переместить веса в папку <i>weights</i>. Достаточно только весов для PyTorch модели. TensorRT веса сбилдятся в процессе. Если возникла ошибка с TensorRT весами, можно удалить *.trt файлы и запустить, тогда они сбилдятся с нуля.

Также в папке <i>data</i> должны быть данные и необходимые файлы для работы. В папке <i>images</i> - картинки,в папке <i>jsons</i> - json файлы с теми же именами что и у картинок.  Примерная структура может быть такой:
```
    - data/

        - images/
            - i.jpg
            ...
        - jsons/
            -i.json
            ...

        - orig_palettes.pickle
        - reference.yaml
    - weights/
    - scripts/
    - config.json
    - main.py
```
```
reference.yaml - файл с формой знака по классам, для работы детектора нарушения целостности
orig_palettes.pickle - файл с оригинальными палитрами, для детектора выцветания
```
Необходимые для работы пути можно заменить в файле <i>config.json</i>

## Установка (Docker)
```
git clone https://gitlab.nanosemantics.ru/na/cv/digital-roads/externaldigitalroads.git
cd externaldigitalroads/
git checkout defectdetector
```
Изменить docker-compose.yml

```
docker-compose build
docker-compose up
```

После выполнения появится папка <i>output</i> c результатами, а также файл <i>all_results.csv</i> в папке <i>output</i> со всеми результатами.

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

После выполнения в папке <i>output</i> появятся файлы в формате json, а также <i>all_results.csv</i> - файл с результатами по всем файлам 

#### Формат выходного json-а

| Поле               | Значение |
|--------------------|---|
| ‘image_name’       | Имя картинки.                 |
| 'discolor_defect'  | True (дефект есть)/False - наличие выцветания знака. Примечание: None - появляется если для данного типа знака нет эталлонной палитры.|
| ‘graffiti_defect’  | True (дефект есть)/False - наличие граффити на знаке.|
| ‘surface_defect’   | True (дефект есть)/False - наличие изменений лицевой поверхности знака. Примечание: None - появляется если знак круглый (в разработке).|