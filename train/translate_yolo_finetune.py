
origin_result = open('result.txt')

target_file = open('result_yolo_rep_channel.csv', 'w')

ostr = origin_result.read()

ostr_splt = ostr.split('test3/data/')

#count_list = ['Compacts', 'Sedans', 'SUVs', 'Coupes', 'Muscle', 'SportsClassics', 'Sports', 'Super', 'Utility', 'Vans', 'Service', 'Emergency', 'Military']

target_file.write('guid/image,N\n')

for idx in range(1, len(ostr_splt)):
    img_detail_split = ostr_splt[idx].split('.jpg')
    img_name = img_detail_split[0]
    img_name = img_name.replace('_', '/')
    car_count = img_detail_split[1].count('car')
    target_file.write(img_name + ',' + str(car_count) + '\n')
