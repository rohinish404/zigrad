# Zigrad

Implementation of Andrej karpathy's [micrograd](https://github.com/karpathy/micrograd) in zig.



## Few pointers for implmenting in zig -
- zig doesnt have set() function so u can use an hashmap with {} (void) value to get almost same functionality. Read [here](https://devlog.hexops.com/2022/zig-hashmaps-explained/)
- In zig, you cannot store pointer fields as keys in hashmaps due to floating point precision or smthing. I needed to store my whole Value struct (which has f64 fields) inside hashmaps. First approach i tried was to create custom hash functions based on data and grad fields.
  So, what happens is when there's no autohashing available, what u do is u need to use [``std.HashMap()``](https://ziglang.org/documentation/master/std/#std.hash_map.HashMap) which takes key value types, a context and a max_load_percentage as parameters. This context is where u define two functions hash and eql (how u want to hash you custom type).
  One issue with the way i did it was if i have the same data value for more than one items, only one will be stored in the hashmap.
  <br />
  So a new solution of hashing i found (thanks to claude also hehe :) ) is to take the memory address of the Value struct and hash them based on that which we can easily get using [``@intToPtr``](https://ziglang.org/documentation/0.7.0/#intToPtr). This is a simple solution that works for now. Will be changing in future if needed.
- 
