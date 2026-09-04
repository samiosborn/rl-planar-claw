# Unified Robot Description Format (URDF) Overview

## Kinematic Tree Rules

URDF represents a robot as a tree of links connected by joints.

Key rules:

* There is one root link with no parent joint.
* Every other link has exactly one parent joint.
* A joint connects exactly one parent link to one child link.
* A parent link can have multiple child joints, allowing branches.
* Closed kinematic loops are not directly representable in standard URDF.
* Link and joint names must be unique.
* Joint origins are defined relative to the parent link frame.
* Child link motion is defined relative to the joint frame.

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

A link has its own coordinate frame, but does not inherently have a size, shape, or mass. These are defined using optional child elements:

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

### Visual

`<visual>` defines how the link is rendered.

It can specify:

* the geometry;
* the geometry's position and orientation relative to the link frame using `<origin>`;
* material and colour.

The main standard geometry types are:

```xml
<geometry>
    <box size="0.10 0.02 0.02"/>
</geometry>
```

`box` uses `size="x y z"` in metres.

```xml
<geometry>
    <cylinder radius="0.02" length="0.10"/>
</geometry>
```

`cylinder` uses:

* `radius` in metres;
* `length` in metres.

The cylinder's axis is along its local `z`-axis.

```xml
<geometry>
    <sphere radius="0.02"/>
</geometry>
```

`sphere` uses `radius` in metres.

```xml
<geometry>
    <mesh filename="path/to/model.stl" scale="1 1 1"/>
</geometry>
```

`mesh` loads geometry from an external file. `scale="x y z"` optionally scales the mesh along each local axis.

The geometry can be translated and rotated relative to the link frame:

```xml
<visual>
    <origin xyz="-0.05 0 0" rpy="0 0 0"/>
    <geometry>
        <box size="0.10 0.02 0.02"/>
    </geometry>
</visual>
```

`xyz` is in metres and `rpy` is roll, pitch, yaw in radians.

Material and colour can also be specified:

```xml
<material name="blue">
    <color rgba="0 0 1 1"/>
</material>
```

`rgba` gives red, green, blue, and alpha values between 0 and 1.

Visual geometry affects rendering only; it does not define collision behaviour or mass properties.

### Collision

`<collision>` defines the geometry to be used by the physics engine for collision and contact detection.

It can differ from the visual geometry. Complex visual meshes are often approximated with simpler collision shapes for faster and more stable simulation. For simple shapes, the collision geometry it is generally the same as the visual geometry. 

Its `<origin>` is also defined relative to the link frame.

### Inertial

`<inertial>` defines the physical mass distribution of the link.

It contains:

* `<mass>`: mass in kilograms (`kg`);
* `<origin>`: centre-of-mass/inertial frame relative to the link frame;
* `<inertia>`: rotational inertia tensor in `kg·m²`.

These properties affect the link's dynamics, including acceleration, gravity, forces, torques, and contact response.

A common convention is to place the link frame at a joint. Primitive geometry such as a box is centred on its own origin, so a link of length `L` extending away from the joint usually has its visual and collision geometry offset by `L / 2`.


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

### Joint Limits

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

For a revolute joint means:

- position range: approximately `-90°` to `+90°`;
- maximum torque: `10 N·m`;
- maximum angular velocity: `2 rad/s`.

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

resisting torque = damping * angular velocity

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

On a revolute joint specifies:

- viscous damping coefficient: `0.1 N·m·s/rad`;
- joint friction torque: `0.05 N·m`.

`<dynamics>` is optional. If it is omitted, standard URDF parsers treat both damping and friction as zero. If `<dynamics>` is present, at least one of `damping` or `friction` must be specified; the omitted value defaults to zero.

Do not confuse joint friction with contact friction between links or objects. Joint friction belongs to `<dynamics>` and resists motion at the joint. Surface/contact friction is a property of collision/contact modelling and is handled separately by the simulator.

#### URDF 1.0/1.1 vs 1.2 Note

In URDF 1.0/1.1, `effort` and `velocity` are required when a `<limit>` element is present. `lower` and `upper` are syntactically optional in the parser and default to `0`, but for a bounded `revolute` or `prismatic` joint they should be specified explicitly; omitting both gives a zero-width position range.

URDF 1.2 changes some joint-limit rules. For `revolute` and `prismatic` joints, `lower` and `upper` are required, while `effort` and `velocity` become optional and default to no finite limit. URDF 1.2 also adds optional `acceleration`, `deceleration`, and `jerk` limits.

## Common Issues

* Distances are specified in metres.
* Angles are specified in radians.
* `<origin>` is relative to another frame, not necessarily the world frame.
* A joint `<origin>` and a visual/collision `<origin>` have different meanings.
* Primitive geometry is centred on its own origin.
* A link extending away from its frame therefore often needs its geometry offset by half its length.
* `<axis>` specifies a direction, not a position.
* `<axis xyz="0 0 0"/>` is invalid because the zero vector has no direction.
* For motion in the `x-y` plane, revolute joints normally rotate around the `z`-axis.
* Visual geometry does not automatically create collision geometry.
* Collision geometry does not define mass or inertia.
* Missing or unrealistic inertial properties can cause unstable or unrealistic dynamics.
* Detailed mesh collision geometry can make simulation slower and less stable than simple primitive shapes.
* Joint friction and damping are defined using `<dynamics>`, not `<limit>`.
* Joint friction is different from surface/contact friction between objects.
* URDF is a tree structure, so closed-loop mechanisms require another modelling approach or simulator-specific constraints.
