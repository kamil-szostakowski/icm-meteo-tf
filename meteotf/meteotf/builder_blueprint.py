from editor import CropArea

accept_fn_index = {
    'precipitation-rain': lambda f: 'R' in f and not 'S' in f,
    'precipitation-snow': lambda f: 'S' in f and not 'R' in f,
    'precipitation-none': lambda f: not 'R' in f and not 'S' in f,
    'precipitation-sleet': lambda f: 'R' in f and 'S' in f,
    'wind-strong': lambda f: 'W' in f,
    'wind-none': lambda f: not 'W' in f,
    'clouds-present': lambda f: 'C' in f,
    'clouds-none': lambda f: not 'C' in f
}

def full_blueprint(intermediate_path):
    precipitation = precipitation_blueprint(intermediate_path)
    wind = wind_blueprint(intermediate_path, 3)
    clouds = clouds_blueprint(intermediate_path, 5)

    return precipitation + wind + clouds

def precipitation_blueprint(intermediate_path, start_label=0):
    return [
        {
            'class_name': 'precipitation-rain',
            'crop_area': CropArea(65, 140, 180, 85),
            'destination_dir': '{root}/precipitation-rain_{label}/'.format(
                root=intermediate_path,
                label=start_label
            ),
            'accept_fn': accept_fn_index['precipitation-rain']            
        },
        {
            'class_name': 'precipitation-snow',
            'crop_area': CropArea(65, 140, 180, 85),
            'destination_dir': '{root}/precipitation-snow_{label}/'.format(
                root=intermediate_path,
                label=start_label + 1
            ),
            'accept_fn': accept_fn_index['precipitation-snow']
        },        
        {
            'class_name': 'precipitation-none',
            'crop_area': CropArea(65, 140, 180, 85),
            'destination_dir': '{root}/precipitation-none_{label}/'.format(
                root=intermediate_path,
                label=start_label + 2
            ),
            'accept_fn': accept_fn_index['precipitation-none']
        }       
    ]

def sleet_blueprint(intermediate_path, start_label=0):
    return [
        {
            'class_name': 'precipitation-sleet',
            'crop_area': CropArea(65, 140, 180, 85),
            'destination_dir': '{root}/precipitation-sleet_{label}/'.format(
                root=intermediate_path,
                label=start_label
            ),
            'accept_fn': accept_fn_index['precipitation-sleet']
        }, 
    ]

def wind_blueprint(intermediate_path, start_label=0):
    return [               
        {
            'class_name': 'wind-strong',
            'crop_area': CropArea(65, 314, 180, 85),
            'destination_dir': '{root}/wind-strong_{label}/'.format(
                root=intermediate_path,
                label=start_label,
            ),
            'accept_fn': accept_fn_index['wind-strong']
        },
        {
            'class_name': 'wind-none',
            'crop_area': CropArea(65, 314, 180, 85),
            'destination_dir': '{root}/wind-none_{label}/'.format(
                root=intermediate_path,
                label=start_label+1,
            ),
            'accept_fn': accept_fn_index['wind-none']
        }       
    ]

def clouds_blueprint(intermediate_path, start_label=0):
    return [
        {
            'class_name': 'clouds-present',
            'crop_area': CropArea(65, 522, 180, 85),
            'destination_dir': '{root}/sky-cloudy_{label}/'.format(
                root=intermediate_path,
                label=start_label,
            ),
            'accept_fn': accept_fn_index['clouds-present']
        },
        {
            'class_name': 'clouds-none',
            'crop_area': CropArea(65, 522, 180, 85),
            'destination_dir': '{root}/sky-clear_{label}/'.format(
                root=intermediate_path,
                label=start_label+1,
            ),
            'accept_fn': accept_fn_index['clouds-none']
        }
    ] 

index = {
    'full': full_blueprint,
    'precipitation': precipitation_blueprint,
    'sleet': sleet_blueprint,
    'wind': wind_blueprint,
    'clouds': clouds_blueprint,
}
    
# {
#     'crop_area': CropArea(65, 140, 180, 85),
#     'destination_dir': '{root}/training-set/storm/'.format(root=intermediate_path),
#     'feature': 'T'
# },