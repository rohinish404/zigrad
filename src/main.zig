const std = @import("std");
const math =  std.math;
const c = @cImport({
    @cInclude("stdlib.h");
    @cInclude("time.h");
});

fn rand_float() f64{
    const random_float:f64 = @as(f64, @floatFromInt(c.rand())) / @as(f64, @floatFromInt(c.RAND_MAX));
    return random_float;
}
pub fn main() !void {
    c.srand(@as(c_uint, @intCast(c.time(null))));
    std.debug.print("Hello world\n", .{});
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();
    // inputs
    var x1 = try Value.init(2.0);
    var x2 =  try Value.init(0.0);
    //weights
    var w1 = try Value.init(-3.0);
    var w2 = try Value.init(1.0);
    //bias
    var b = try Value.init(6.7);

    var x1w1 = try x1.mult(&w1);
    var x2w2 = try x2.mult(&w2);
    var x1w1x2w2 = try x1w1.add(&x2w2);

    var n = try x1w1x2w2.add(&b);

    var o = try n.relu();

    std.debug.print("data = {d:.2}, grad = {d:.2}, children = {any}\n", .{o.data, o.grad, o.children}); 
    std.debug.print("-----------------------\n", .{});

    try o.backward(allocator);
    std.debug.print("data = {d:.2}, grad = {d:.2}, children = {any}\n", .{n.data, n.grad, n.children});

    std.debug.print("------------------------------------------------------------\n", .{});
    var arr = [_]u64{4,4,1};
    var mlp = try MLP.init(allocator, 2, arr[0..]);
    defer mlp.deinit();


    var arr2 = [_]f64{2.0,3.0, -1.0};
    const res = try mlp.call(arr2[0..]);
    defer res.deinit();
    std.debug.print("res - {d:.2}", .{res.items[0].data});

}


const Neuron = struct {
    weights: std.ArrayList(Value),
    bias: Value,
    allocator: std.mem.Allocator,
    pub fn init(allocator: std.mem.Allocator, nin: u64) !Neuron {
        var weights = std.ArrayList(Value).init(allocator);
        const bias = try Value.init(rand_float());
        for (0..nin) |_| {
            const random_weight = rand_float();
            try weights.append(try Value.init(random_weight));
        }
        return .{
            .weights = weights,
            .allocator = allocator,
            .bias = bias,
        };
    }
    fn call(self: @This(), x: []Value) !Value {
        var result = try Value.init(0);
        for (self.weights.items, 0..) |weight, i| {
            var product = try @constCast(&weight).mult(&x[i]);
            result = try result.add(&product);
        }
        result = try result.add(@constCast(&self.bias));
        return try result.relu();
    }

    //fn paramters(self: @This()) void{}


    pub fn deinit(self: *Neuron) void {

        self.weights.deinit();
    }


};

const Layer = struct {
    neurons: std.ArrayList(Neuron),
    allocator: std.mem.Allocator,
    pub fn init(allocator: std.mem.Allocator, nin: u64, nout: u64) !Layer {
        var neurons = std.ArrayList(Neuron).init(allocator);
        for (0..nout) |_| {
            try neurons.append(try Neuron.init(allocator, nin));
        }
        return .{
            .neurons = neurons,
            .allocator = allocator,
        };
    }

    fn call(self: @This(), x: []Value) !std.ArrayList(Value) {
        var final = std.ArrayList(Value).init(self.allocator);
        errdefer final.deinit();
        for (self.neurons.items) |neuron| {
            try final.append(try neuron.call(x));
        }
        return final;
    }

    //fn paramters(self: @This()) void{}


    pub fn deinit(self: *Layer) void {
        for (self.neurons.items) |*neuron| {
            neuron.deinit();
        }
        self.neurons.deinit();
    }
};

const MLP = struct {
    sz: std.ArrayList(u64),
    layers: std.ArrayList(Layer),
    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator, nin:u64, nout: []const u64) !MLP {
        var sz = std.ArrayList(u64).init(allocator);
        errdefer sz.deinit();
        try sz.append(nin);
        for (nout) |out| {
            try sz.append(out);
        }
        var layers = std.ArrayList(Layer).init(allocator);
        errdefer layers.deinit();
        for (nout, 0..) |_,i| {
            try layers.append(try Layer.init(allocator, sz.items[i], sz.items[i+1]));
        }
        return .{
            .sz=sz,
            .layers=layers,
            .allocator=allocator,
        };

    }
    fn call(self: @This(), x: []f64) !std.ArrayList(Value){
        var input = std.ArrayList(Value).init(self.allocator);
        errdefer input.deinit();

        for (x) |val| {
            try input.append(try Value.init(val));
        }

        for (self.layers.items, 0..) |layer, i| {
            var new_input = try layer.call(input.items);
            std.debug.print("Layer {d} output: {any}\n", .{i, new_input.items});
            new_input.capacity += 0;
            input.deinit(); 
            input = new_input; 
            std.debug.print("---------------------------------=-----------------------", .{});
        }

        return input;
    }    

    pub fn deinit(self: *MLP) void {
        for (self.layers.items) |*layer| {
            layer.deinit();
        }
        self.sz.deinit();
        self.layers.deinit();
    }


};




const Value = struct {
    data: f64,
    grad: f64,
    children: ?struct {
        self: *Value,
        other: ?*Value,
    },
    backward_fn: ?*const fn (*Value, ?*Value, *Value) void,

    pub fn init(data: f64) !Value{
        return .{
            .data = data,
            .grad = 0.0,
            .children = null,
            .backward_fn = null,
        };

    }
    fn add(self: *Value, other: *Value) !Value{
        var result = try Value.init(self.data + other.data);
        result.children = .{.self=self,.other=other};
        const _backward = struct {
            fn backward(first: *Value, sec:?*Value, out: *Value) void {
                first.grad += out.grad;
                if (sec) |second| {
                    second.grad += out.grad;
                }
            }
        }.backward;

        result.backward_fn = _backward;
        return result;

    }

    fn mult(self: *Value, other: *Value) !Value{
        var result = try Value.init(self.data * other.data);
        result.children = .{.self=self,.other=other};

        const _backward = struct {
            fn backward(first: *Value, sec:?*Value, out: *Value) void {
                if (sec) |second| {
                    first.grad += out.grad*second.data;
                    second.grad += out.grad*first.data;
                }
            }
        }.backward;

        result.backward_fn = _backward;
        return result;    
    }
    fn tanh(self: *Value) !Value{
        const res: f64 = (math.exp(2*self.data) - 1)/(math.exp(2*self.data) + 1);
        var result = try Value.init(res);
        result.children = .{.self=self, .other=null};
        return result;    
    }

    fn pow(self: *Value, other: f64) !Value{
        var result = try Value.init(math.pow(f64, self.data, other));
        result.children = .{.self=self, .other=null};

        const _backward = struct {
            exponent:f64,
            fn backward(first: *Value, out: *Value) void {
                first.grad+= ((@This().exponent)*math.pow(f64, first.data, @This().exponent - 1)) * out.grad;
            }
        }.backward;
        result.backward_fn = _backward;
        return result;    

    }

    fn relu(self: *Value) !Value{
        const res: f64 = if (self.data < 0) 0 else self.data;

        var result = try Value.init(res);
        result.children = .{.self=self, .other=null};
        const _backward = struct {
            fn backward(first: *Value, sec: ?*Value, out: *Value) void {
                if (sec == null){
                    if (out.grad > 0) {
                        first.grad += 1 * out.grad;
                    }else {
                        first.grad += 0;
                    }
                }}
        }.backward;
        result.backward_fn = _backward;
        return result;    


    }

    fn backward(self: *Value, allocator: std.mem.Allocator) !void {
        var topo= std.ArrayList(*Value).init(allocator);
        defer topo.deinit();
        var visited = std.HashMap(*Value, void, MyKeyContext, std.hash_map.default_max_load_percentage).init(allocator);
        defer visited.deinit();
        try build_topo(self, &topo, &visited);

        std.debug.print("{any}\n", .{topo.items.len});
        std.debug.print("-----------------------------------\n", .{});
        self.grad = 1.0;

        var i: usize = topo.items.len;
        while (i > 0) {
            i -= 1;
            const v = topo.items[i];
            if (v.backward_fn) |backward_fn| {
                if (v.children) |children| {
                    backward_fn(children.self,children.other orelse null,v);
                }
            }

        }

    }
    fn build_topo(v:*Value, topo: *std.ArrayList(*Value),visited: *std.HashMap(*Value,void,MyKeyContext,std.hash_map.default_max_load_percentage)) !void {
        if (!visited.contains(v)){
            try visited.put(v, {});
            if (v.children) |children|{
                try build_topo(children.self, topo, visited);
                if (children.other) |other|{
                    try build_topo(other, topo, visited);
                }
            }
            try topo.append(v);
        }
    }
    const MyKeyContext = struct {
        pub fn hash(_: @This(), x: *Value) u64 {
            return @intFromPtr(x);
        }

        pub fn eql(_: @This(), a: *Value, b: *Value) bool {
            return a == b;
        }
    };
};



