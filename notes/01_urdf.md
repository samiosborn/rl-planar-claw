# Unified Robot Description Format (URDF) Overview

## File Structure

A URDF is an XML file describing a robot as rigid links connected by joints.

The entire model is contained inside a `<robot>` element:

```xml
<robot name="planar_claw">

    <link name="base"/>

    <link name="link_1"/>

    <joint name="joint_1" type="revolute">
        <parent link="base"/>
        <child link="link_1"/>
    </joint>

</robot>
```

The main elements are:

- `<robot>`: contains the complete model.
- `<link>`: defines a rigid body.
- `<joint>`: defines the relationship and allowed motion between two links.

Links and joints are referenced by unique names.

If the `<robot>` element does not specify a URDF version, standard `urdfdom` interprets the file as URDF 1.0.

## Links

A `<link>` represents a rigid body. Everything belonging to the same link moves together.

```xml
<link name="link_1">
</link>
```

A link has its own coordinate frame, but does not inherently have a size or shape. Geometry and physical properties are defined inside the link:

```xml
<link name="link_1">

    <visual>
    </visual>

    <collision>
    </collision>

    <inertial>
    </inertial>

</link>
```

- `<visual>` defines the geometry used for rendering.
- `<collision>` defines the geometry used for collision and contact.
- `<inertial>` defines mass, centre of mass, and rotational inertia.

These elements can have different geometries and origins.

A common convention is to place the link frame at a joint. Primitive geometry such as a box is centred on its own origin, so a link of length `L` extending away from the joint usually needs its visual and collision geometry offset by `L / 2`.

The link is the rigid body; its visual geometry is only a representation of that body.

## Joints

A `<joint>` connects one parent link to one child link and defines how the child can move relative to the parent.

The joint `name` and `type` attributes are required.

```xml
<joint name="joint_1" type="revolute">
    <parent link="base"/>
    <child link="link_1"/>
</joint>
```

`<parent>` and `<child>` are required.

Standard joint types:

- `fixed`: no relative motion.
- `revolute`: rotation around one axis with lower and upper angular limits.
- `continuous`: rotation around one axis without angular position limits.
- `prismatic`: translation along one axis with lower and upper position limits.
- `planar`: motion within a plane.
- `floating`: unrestricted 6-DoF translation and rotation.

A movable joint can additionally define:

```xml
<origin xyz="0 0 0" rpy="0 0 0"/>
<axis xyz="0 0 1"/>
<limit lower="-1.57" upper="1.57" effort="10" velocity="2"/>
<dynamics damping="0.1" friction="0.05"/>
```

- `<origin>` is optional. It defines the joint frame relative to the parent link frame. If omitted, the transform is zero translation and zero rotation.
- `<axis>` is used by joints with a single motion axis. If omitted, the URDF default axis is `(1, 0, 0)`. Fixed and floating joints do not use an axis.
- `<limit>` defines hard joint position limits and actuator torque/force and velocity limits.
- `<dynamics>` is optional and defines passive joint damping and friction.

For a revolute joint, the child link rotates around the joint axis.

## Coordinate Frames and Origins

## Joint Axes

## Visual vs Collision vs Inertial

## Joint Limits

For the URDF 1.0 syntax used when no `version` is specified on `<robot>`, a `revolute` or `prismatic` joint requires a `<limit>` element.

Example for a revolute joint:

```xml
<limit lower="-1.57" upper="1.57" effort="10" velocity="2"/>
```

The attributes mean:

- `lower`: minimum joint position.
  - Revolute joint: radians (`rad`).
  - Prismatic joint: metres (`m`).
- `upper`: maximum joint position.
  - Revolute joint: radians (`rad`).
  - Prismatic joint: metres (`m`).
- `effort`: maximum actuator output allowed at the joint.
  - Revolute or continuous joint: maximum torque in newton-metres (`N·m`).
  - Prismatic joint: maximum force in newtons (`N`).
- `velocity`: maximum magnitude of joint velocity.
  - Revolute or continuous joint: radians per second (`rad/s`).
  - Prismatic joint: metres per second (`m/s`).

For example:

```xml
<limit lower="-1.57" upper="1.57" effort="10" velocity="2"/>
```

for a revolute joint means:

- position range: approximately `-90°` to `+90°`;
- maximum torque: `10 N·m`;
- maximum angular velocity: `2 rad/s`.

In URDF 1.0/1.1, `effort` and `velocity` are required when a `<limit>` element is present. `lower` and `upper` are syntactically optional in the parser and default to `0`, but for a bounded `revolute` or `prismatic` joint they should be specified explicitly; omitting both gives a zero-width position range.

A `continuous` joint has no lower or upper angular position limit. It may still specify torque and velocity limits:

```xml
<joint name="wheel_joint" type="continuous">
    <parent link="base"/>
    <child link="wheel"/>
    <axis xyz="0 0 1"/>
    <limit effort="10" velocity="5"/>
</joint>
```

`fixed` joints do not move and therefore do not need motion limits.

### Joint Dynamics

Damping and friction are not part of `<limit>`. They are defined by the optional `<dynamics>` element:

```xml
<dynamics damping="0.1" friction="0.05"/>
```

`damping` models a resisting force or torque proportional to joint velocity.

For a revolute joint:

```text
resisting torque = damping × angular velocity
```

Units:

- revolute or continuous joint damping: `N·m·s/rad`;
- prismatic joint damping: `N·s/m`.

`friction` models joint friction that resists motion.

Units:

- revolute or continuous joint friction: `N·m`;
- prismatic joint friction: `N`.

For example:

```xml
<dynamics damping="0.1" friction="0.05"/>
```

on a revolute joint specifies:

- viscous damping coefficient: `0.1 N·m·s/rad`;
- joint friction torque: `0.05 N·m`.

`<dynamics>` is optional. If it is omitted, standard URDF parsers treat both damping and friction as zero. If `<dynamics>` is present, at least one of `damping` or `friction` must be specified; the omitted value defaults to zero.

Do not confuse joint friction with contact friction between links or objects. Joint friction belongs to `<dynamics>` and resists motion at the joint. Surface/contact friction is a property of collision/contact modelling and is handled separately by the simulator.

### URDF 1.2 Note

URDF 1.2 changes some joint-limit rules. For `revolute` and `prismatic` joints, `lower` and `upper` are required, while `effort` and `velocity` become optional and default to no finite limit. URDF 1.2 also adds optional `acceleration`, `deceleration`, and `jerk` limits.

## Kinematic Tree Rules

## Common Issues

## Worked Example
