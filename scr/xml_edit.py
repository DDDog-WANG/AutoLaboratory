
xml_string = """
<mesh file="pipette_collision_0.obj"/>
<mesh file="pipette_collision_1.obj"/>
<mesh file="pipette_collision_2.obj"/>
<mesh file="pipette_collision_3.obj"/>
<mesh file="pipette_collision_4.obj"/>
<mesh file="pipette_collision_5.obj"/>
<mesh file="pipette_collision_6.obj"/>
<mesh file="pipette_collision_7.obj"/>
<mesh file="pipette_collision_8.obj"/>
<mesh file="pipette_collision_9.obj"/>
<mesh file="pipette_collision_10.obj"/>
<mesh file="pipette_collision_11.obj"/>
<mesh file="pipette_collision_12.obj"/>
<mesh file="pipette_collision_13.obj"/>
<mesh file="pipette_collision_14.obj"/>
<mesh file="pipette_collision_15.obj"/>
<mesh file="pipette_collision_16.obj"/>
<mesh file="pipette_collision_17.obj"/>
<mesh file="pipette_collision_18.obj"/>
<mesh file="pipette_collision_19.obj"/>
<mesh file="pipette_collision_20.obj"/>
<mesh file="pipette_collision_21.obj"/>
<mesh file="pipette_collision_22.obj"/>
<mesh file="pipette_collision_23.obj"/>
<mesh file="pipette_collision_24.obj"/>
<mesh file="pipette_collision_25.obj"/>
<mesh file="pipette_collision_26.obj"/>
<mesh file="pipette_collision_27.obj"/>
<mesh file="pipette_collision_28.obj"/>
<mesh file="pipette_collision_29.obj"/>
<mesh file="pipette_collision_30.obj"/>
<mesh file="pipette_collision_31.obj"/>
<mesh file="pipette_collision_32.obj"/>
<mesh file="pipette_collision_33.obj"/>
<mesh file="pipette_collision_34.obj"/>
<mesh file="pipette_collision_35.obj"/>
<mesh file="pipette_collision_36.obj"/>
<mesh file="pipette_collision_37.obj"/>
<mesh file="pipette_collision_38.obj"/>
<mesh file="pipette_collision_39.obj"/>
<mesh file="pipette_collision_40.obj"/>
<mesh file="pipette_collision_41.obj"/>
<mesh file="pipette_collision_42.obj"/>
<mesh file="pipette_collision_43.obj"/>
<mesh file="pipette_collision_44.obj"/>
<mesh file="pipette_collision_45.obj"/>
<mesh file="pipette_collision_46.obj"/>
<mesh file="pipette_collision_47.obj"/>
<mesh file="pipette_collision_48.obj"/>
<mesh file="pipette_collision_49.obj"/>
<mesh file="pipette_collision_50.obj"/>
<mesh file="pipette_collision_51.obj"/>
<mesh file="pipette_collision_52.obj"/>
<mesh file="pipette_collision_53.obj"/>
<mesh file="pipette_collision_54.obj"/>
<mesh file="pipette_collision_55.obj"/>
<mesh file="pipette_collision_56.obj"/>
<mesh file="pipette_collision_57.obj"/>
<mesh file="pipette_collision_58.obj"/>
<mesh file="pipette_collision_59.obj"/>
<mesh file="pipette_collision_60.obj"/>
<mesh file="pipette_collision_61.obj"/>
<mesh file="pipette_collision_62.obj"/>
<mesh file="pipette_collision_63.obj"/>
<mesh file="pipette_collision_64.obj"/>
<mesh file="pipette_collision_65.obj"/>
<mesh file="pipette_collision_66.obj"/>
<mesh file="pipette_collision_67.obj"/>
<mesh file="pipette_collision_68.obj"/>
<mesh file="pipette_collision_69.obj"/>
<mesh file="pipette_collision_70.obj"/>
<mesh file="pipette_collision_71.obj"/>
<mesh file="pipette_collision_72.obj"/>
<mesh file="pipette_collision_73.obj"/>
<mesh file="pipette_collision_74.obj"/>
<mesh file="pipette_collision_75.obj"/>
<mesh file="pipette_collision_76.obj"/>
<mesh file="pipette_collision_77.obj"/>
<mesh file="pipette_collision_78.obj"/>
<mesh file="pipette_collision_79.obj"/>
<mesh file="pipette_collision_80.obj"/>
<mesh file="pipette_collision_81.obj"/>
<mesh file="pipette_collision_82.obj"/>
<mesh file="pipette_collision_83.obj"/>
<mesh file="pipette_collision_84.obj"/>
<mesh file="pipette_collision_85.obj"/>
<mesh file="pipette_collision_86.obj"/>
<mesh file="pipette_collision_87.obj"/>
<mesh file="pipette_collision_88.obj"/>
<mesh file="pipette_collision_89.obj"/>
<mesh file="pipette_collision_90.obj"/>
<mesh file="pipette_collision_91.obj"/>
<mesh file="pipette_collision_92.obj"/>
<mesh file="pipette_collision_93.obj"/>
<mesh file="pipette_collision_94.obj"/>
<mesh file="pipette_collision_95.obj"/>
<mesh file="pipette_collision_96.obj"/>
<mesh file="pipette_collision_97.obj"/>
<mesh file="pipette_collision_98.obj"/>
<mesh file="pipette_collision_99.obj"/>
<mesh file="pipette_collision_100.obj"/>
<mesh file="pipette_collision_101.obj"/>
<mesh file="pipette_collision_102.obj"/>
<mesh file="pipette_collision_103.obj"/>
<mesh file="pipette_collision_104.obj"/>
<mesh file="pipette_collision_105.obj"/>
<mesh file="pipette_collision_106.obj"/>
<mesh file="pipette_collision_107.obj"/>
<mesh file="pipette_collision_108.obj"/>
<mesh file="pipette_collision_109.obj"/>
<mesh file="pipette_collision_110.obj"/>
<mesh file="pipette_collision_111.obj"/>
<mesh file="pipette_collision_112.obj"/>
<mesh file="pipette_collision_113.obj"/>
<mesh file="pipette_collision_114.obj"/>
<mesh file="pipette_collision_115.obj"/>
<mesh file="pipette_collision_116.obj"/>
<mesh file="pipette_collision_117.obj"/>
<mesh file="pipette_collision_118.obj"/>
<mesh file="pipette_collision_119.obj"/>
<mesh file="pipette_collision_120.obj"/>
<mesh file="pipette_collision_121.obj"/>
<mesh file="pipette_collision_122.obj"/>
<mesh file="pipette_collision_123.obj"/>
<mesh file="pipette_collision_124.obj"/>
<mesh file="pipette_collision_125.obj"/>
<mesh file="pipette_collision_126.obj"/>
<mesh file="pipette_collision_127.obj"/>
"""



xml_lines = xml_string.split('\n')

new_xml_lines = []
for line in xml_lines:
    if line.strip():  # 忽略空行
        # 从文件名中提取数字
        number = line.split('_')[-1].split('.')[0]
        # 插入name属性
        new_line = line.replace(f'file="pipette_collision_{number}.obj"', f'name="pipette_collision_{number}" file="meshes/pipette/pipette_collision_{number}.obj" scale="0.001 0.001 0.001"')
        new_xml_lines.append(new_line)

# 将新行连接成一个字符串
new_xml_string = '\n'.join(new_xml_lines)

print(new_xml_string)

