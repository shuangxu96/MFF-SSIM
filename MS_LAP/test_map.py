# -*- coding: utf-8 -*-
"""
Created on Tue Dec 17 21:12:34 2019

@author: win10
"""

from MFF import MFF, load_images

''' ===== coffee ====== '''
x = load_images('./test_image/coffee').cuda()

# gfdf
model1 = MFF(input = x, map_mode = 'gfdf')
model1.train(max_iter = 1000)
model1.show_curve()
model1.show_image()
model1.save_image('./result_different_maps/coffee/gfdf.png')
model1.save_map('./result_different_maps/coffee/gfdf_map.png')

# var
model2 = MFF(input = x, map_mode = 'var')
model2.train(max_iter = 1)
model2.show_curve()
model2.show_image()
model2.save_image('./result_different_maps/coffee/var.png')
model2.save_map('./result_different_maps/coffee/var_map.png')

# lap
model3 = MFF(input = x, map_mode = 'lap')
model3.train(max_iter = 1000)
model3.show_curve()
model3.show_image()
model3.save_image('./result_different_maps/coffee/lap.png')
model3.save_map('./result_different_maps/coffee/lap_map.png')


''' ===== shell ====== '''
x = load_images('./test_image/shell').cuda()

# gfdf # [Warning: gfdf does not supoort more than two images!]
#model4 = MFF(input = x, map_mode = 'gfdf')
#model4.train(max_iter = 1000)
#model4.show_curve()
#model4.show_image()
#model4.save_image('./result_different_maps/shell/gfdf.png')
#model4.save_map('./result_different_maps/shell/gfdf_map.png')

# var
model5 = MFF(input = x, map_mode = 'var')
model5.train(max_iter = 1000)
model5.show_curve()
model5.show_image()
model5.save_image('./result_different_maps/shell/var.png')
model5.save_map('./result_different_maps/shell/var_map.png')

# lap
model6 = MFF(input = x, map_mode = 'lap')
model6.train(max_iter = 1000)
model6.show_curve()
model6.show_image()
model6.save_image('./result_different_maps/shell/lap.png')
model6.save_map('./result_different_maps/shell/lap_map.png')




