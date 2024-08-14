const std = @import("std");
const testing = std.testing;
const Engine = @import("engine.zig");
const Value = Engine.Value;

test "Value initialization" {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    const v = try Value.init(allocator, 5.0);
    defer v.deinit();

    try testing.expectEqual(v.data, 5.0);
    try testing.expectEqual(v.grad, 0.0);
    try testing.expect(v.children == null);
    try testing.expect(v.backward_fn == null);
}

test "Value addition" {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    const a = try Value.init(allocator, 2.0);
    defer a.deinit();
    const b = try Value.init(allocator, 3.0);
    defer b.deinit();

    const c = try a.add(b);
    defer c.deinit();

    try testing.expectEqual(c.data, 5.0);
    try testing.expect(c.children != null);
    try testing.expect(c.backward_fn != null);
}

test "Value multiplication" {
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    const allocator = arena.allocator();

    const a = try Value.init(allocator, 2.0);
    defer a.deinit();
    const b = try Value.init(allocator, 3.0);
    defer b.deinit();

    const c = try a.mult(b);
    defer c.deinit();

    try testing.expectEqual(c.data, 6.0);
    try testing.expect(c.children != null);
    try testing.expect(c.backward_fn != null);
}

test "Value backward pass" {
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    const allocator = arena.allocator();

    const a = try Value.init(allocator, 2.0);
    defer a.deinit();
    const b = try Value.init(allocator, 3.0);
    defer b.deinit();

    const c = try a.mult(b);
    defer c.deinit();

    try c.backward();

    try testing.expectEqual(a.grad, 3.0);
    try testing.expectEqual(b.grad, 2.0);
}
