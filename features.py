from django.shortcuts import render

def predict_view(request):
    display_names = {
        'radius_mean': 'Mean Radius',
        'texture_mean': 'Mean Texture',
        'perimeter_mean': 'Mean Perimeter',
        'area_mean': 'Mean Area',
        'smoothness_mean': 'Mean Smoothness',
        'compactness_mean': 'Mean Compactness',
        'concavity_mean': 'Mean Concavity',
        'concave points_mean': 'Mean Concave Points',
        'symmetry_mean': 'Mean Symmetry',
        'fractal_dimension_mean': 'Mean Fractal Dimension',
        'radius_se': 'SE Radius',
        'texture_se': 'SE Texture',
        'perimeter_se': 'SE Perimeter',
        'area_se': 'SE Area',
        'smoothness_se': 'SE Smoothness',
        'compactness_se': 'SE Compactness',
        'concavity_se': 'SE Concavity',
        'concave points_se': 'SE Concave Points',
        'symmetry_se': 'SE Symmetry',
        'fractal_dimension_se': 'SE Fractal Dimension',
        'radius_worst': 'Worst Radius',
        'texture_worst': 'Worst Texture',
        'perimeter_worst': 'Worst Perimeter',
        'area_worst': 'Worst Area',
        'smoothness_worst': 'Worst Smoothness',
        'compactness_worst': 'Worst Compactness',
        'concavity_worst': 'Worst Concavity',
        'concave points_worst': 'Worst Concave Points',
        'symmetry_worst': 'Worst Symmetry',
        'fractal_dimension_worst': 'Worst Fractal Dimension'
    }
    
    return render(request, 'predict.html', {'display_names': display_names})

    