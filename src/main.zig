const std = @import("std");
const math =  std.math;

pub fn main() !void {
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

    var o = try x1w1x2w2.add(&b);

    //var o = try n.tanh();

    std.debug.print("-----------------------\n", .{});

    try o.backward(allocator);
    std.debug.print("data = {d:.2}, grad = {d:.2}, children = {any}\n", .{x1.data, x1.grad, x1.children}); 
}

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



