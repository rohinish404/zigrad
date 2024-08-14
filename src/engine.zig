const std = @import("std");
const math = std.math;

pub const Value = struct {
    data: f64,
    grad: f64,
    children: ?*Children,
    backward_fn: ?*const fn (*Value, *Children, *Value) void,
    allocator: std.mem.Allocator,
    extra_data: ?f64,
    
    const Children = struct {
        self: *Value,
        other: ?*Value,
    };

    pub fn init(allocator: std.mem.Allocator, data: f64) !*Value {
        const self = try allocator.create(Value);
        errdefer allocator.destroy(self);
        self.* = .{
            .data = data,
            .grad = 0.0,
            .children = null,
            .backward_fn = null,
            .allocator = allocator,
            .extra_data = null,
        };
        return self;
    }

    pub fn deinit(self: *Value) void {
        if (self.children) |children| {
            self.allocator.destroy(children);
        }
        self.allocator.destroy(self);
    }
    pub fn add(self: *Value, other: *Value) !*Value{
        var result = try Value.init(self.allocator, self.data + other.data);
        errdefer result.deinit();
        result.children = try self.allocator.create(Children);
        errdefer self.allocator.destroy(result.children.?);
        result.children.?.* = .{ .self = self, .other = other };

        const _backward = struct {
            fn backward(_: *Value, children: *Children, out: *Value) void {
                children.self.grad += out.grad;
                if (children.other) |second| {
                    second.grad += out.grad;
                }
            }
        }.backward;
        result.backward_fn = _backward;
        return result;

    }
    pub fn sub(self: *Value, other: *Value) !*Value{
        var result = try Value.init(self.allocator, self.data - other.data);
        errdefer result.deinit();
        result.children = try self.allocator.create(Children);
        errdefer self.allocator.destroy(result.children.?);
        result.children.?.* = .{ .self = self, .other = other };

        const _backward = struct {
            fn backward(_: *Value, children: *Children, out: *Value) void {
                children.self.grad += out.grad;
                if (children.other) |second| {
                    second.grad -= out.grad;
                }
            }
        }.backward;
        result.backward_fn = _backward;
        return result;

    }
    pub fn negate(self: *Value) !*Value {
        var result = try Value.init(self.allocator, -self.data);
        errdefer result.deinit();
        return result;
    }


    pub fn mult(self: *Value, other: *Value) !*Value{
        var result = try Value.init(self.allocator, self.data * other.data);
        errdefer result.deinit();
        result.children = try self.allocator.create(Children);
        errdefer self.allocator.destroy(result.children.?);
        result.children.?.* = .{ .self = self, .other = other };

        const _backward = struct {
            fn backward(_: *Value,children: *Children, out: *Value) void {
                children.self.grad += out.grad * children.other.?.data;
                children.other.?.grad += out.grad * children.self.data;
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

    pub fn pow(self: *Value, exponent: f64) !*Value {
        var result = try Value.init(self.allocator, math.pow(f64, self.data, exponent));
        errdefer result.deinit();
        result.children = try self.allocator.create(Children);
        result.children.?.* = .{ .self = self, .other = null };
        result.extra_data = exponent;

        const _backward = struct {
            fn backward(_: *Value,children: *Children, out: *Value) void {
                const exp = children.self.extra_data;
                children.self.grad += ((exp.?) * math.pow(f64, children.self.data, (exp.?) - 1.0)) * out.grad;
            }
        }.backward;

        result.backward_fn = _backward;
        return result;  
    }


    pub fn relu(self: *Value) !*Value {
        const res: f64 = if (self.data < 0) 0 else self.data;
        var result = try Value.init(self.allocator, res);
        errdefer result.deinit();
        result.children = try self.allocator.create(Children);
        result.children.?.* = .{ .self = self, .other = null };

        const _backward = struct {
            fn backward(_: *Value,children: *Children, out: *Value) void {
                if (out.data > 0) {
                    children.self.grad += out.grad;
                }
            }
        }.backward;

        result.backward_fn = _backward;
        return result;
    }


    pub fn backward(self: *Value) !void {
        var topo = std.ArrayList(*Value).init(self.allocator);
        defer topo.deinit();
        var visited = std.HashMap(*Value, void, MyKeyContext, std.hash_map.default_max_load_percentage).init(self.allocator);
        defer visited.deinit();
        try build_topo(self, &topo, &visited);

        self.grad = 1.0;
        std.debug.print("Starting backward pass\n", .{});
        std.debug.print("Topological sort size: {}\n", .{topo.items.len});
        var i: usize = topo.items.len;
        while (i > 0) {
            i -= 1;
            const v = topo.items[i];
            std.debug.print("Processing node {}: data={d}, grad={d}\n", .{i, v.data, v.grad});

            if (v.backward_fn) |backward_fn| {
                if (v.children) |children| {
                    backward_fn(v, children, v);
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


