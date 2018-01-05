import PyDeepCL

last = None
# if use seed, should always be same value:
print('with seed:')
for it in range(5):
    PyDeepCL.RandomSingleton.seed(123)
    val = PyDeepCL.RandomSingleton.uniform()
    print('', it, val)
    if last is None:
        last = val
    assert last == val

# if dont use seed should always be different:
print('without seed:')
last = None
for it in range(5):
    val = PyDeepCL.RandomSingleton.uniform()
    print('', it, val)
    assert last != val
    last = val

