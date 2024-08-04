const std = @import("std");

pub fn main() !void {
    std.debug.print("Hello world\n", .{});
//    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
//    defer _ = gpa.deinit();
//    const allocator = gpa.allocator();
//
    var a = try Value.init(2.0);
    var b =  try Value.init(3.0);
    var result = try a.mult(&b);
    const c = try result.add(&a);
    std.debug.print("data = {d:.2}, grad = {d:.2}, children = {any}\n", .{c.data, c.grad, c.children}); 
}

const Value = struct {
    data: f64,
    grad: f64,
    children: ?struct {
        self: *Value,
        other: *Value,
    },
    pub fn init(data: f64) !Value{
        return .{
            .data = data,
            .grad = 0.0,
            .children = null,
        };

    }
    fn add(self: *Value, other: *Value) !Value{
        var result = try Value.init(self.data + other.data);
        result.children = .{.self=self,.other=other};
        return result;    
    }

    fn mult(self: *Value, other: *Value) !Value{
        var result = try Value.init(self.data * other.data);
        result.children = .{.self=self,.other=other};
        return result;    
    }
};



