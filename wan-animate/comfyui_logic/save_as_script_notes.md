# Workflow Edits

## 1. Replace has_manager logic

Before:
```python
if os.path.isdir(manager_path) and os.listdir(manager_path):
    sys.path.append(manager_path)
    global has_manager
    has_manager = True
```

After:
```python
global has_manager
if os.path.isdir(manager_path) and os.listdir(manager_path):
    sys.path.append(manager_path)
    has_manager = True
else:
    has_manager = False
````

---

## 2. Replace Parent directory check logic

Before:
```python
if parent_directory == path:
    return None
```

After:
```python
if parent_directory == path or not parent_directory:
    return None
```

---

## 3. Wrap `easy_showanything` input in list (all 3 occurrences)

Before:

```python
easy_showanything_147 = easy_showanything.log_input(
    text="110", anything=get_value_at_index(mathexpressionpysssss_144, 0)
)
```

After:

```python
easy_showanything_147 = easy_showanything.log_input(
    text="110", anything=[get_value_at_index(mathexpressionpysssss_144, 0)]
)
```

*(Apply the same change to the other two `easy_showanything` occurrences.)*

---

## 4. Double check if constants in workflow match desired constants for input/output, config and models


