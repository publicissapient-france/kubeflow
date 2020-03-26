it - plusieurs opérations simples d'addition. La tâche 1 et la tâche 2 affichent certaines constantes aux paramètres d'entrée, puis la tâche 3 ajuts la sortie de la tâche 1 à la sortie de la tâche 2.   

import kfp
import kfp.dsl as dsl
from kfp import compiler
from kfp import components

BASE_IMAGE = 'python:3.6-slim'

dsl.python_component(
    name='addition_op',
    description='sums up two numbers',
    base_image=BASE_IMAGE 
)
def summa(a: float, b: float) -> float:
    '''Calculates sum of two arguments'''
    return a + b

add_op = components.func_to_container_op(
    summa,
    base_image=BASE_IMAGE, 
)
@dsl.pipeline(
   name='Addition pipeline',
   description='A simle test pipeline.'
)
def calc_pipeline(
   a: float = 1,
   b: float = 2
):
    add_1_task = add_op(a, 10) 
    add_2_task = add_op(5, b)
    add_3_task = add_op(add_1_task.output, add_2_task.output)

pipeline_func = calc_pipeline
pipeline_filename = 'test_pipeline.yaml'
compiler.Compiler().compile(pipeline_func, pipeline_filename)
