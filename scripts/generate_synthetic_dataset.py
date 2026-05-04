"""Generate humanoid egocentric perception data with Isaac Sim Replicator.

Run this script with Isaac Sim's Python interpreter:

    /path/to/isaac-sim/python.sh scripts/generate_synthetic_dataset.py

The scene approximates a head-mounted humanoid camera observing manipulation
targets in a work area. Replace primitives with robot/workcell USD assets as
the next iteration.
"""

from __future__ import annotations

import argparse
import os
import random
import shutil
from pathlib import Path
from typing import Any

import yaml


def log(message: str) -> None:
    print(f"[dataset-generator] {message}", flush=True)


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as config_file:
        return yaml.safe_load(config_file)


def rgb255(red: int, green: int, blue: int) -> tuple[float, float, float]:
    """Convert 8-bit RGB values to the normalized range expected by OmniPBR."""
    return (red / 255.0, green / 255.0, blue / 255.0)


def create_visible_material(
    rep: Any,
    color: tuple[float, float, float],
    roughness: float,
) -> Any:
    """Create a material that remains visible in headless Replicator RGB output."""
    return rep.create.material_omnipbr(
        diffuse=color,
        roughness=roughness,
        emissive_color=color,
        emissive_intensity=255.0,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/dataset.yaml"),
        help="Dataset generation config path.",
    )
    parser.add_argument(
        "--gui",
        action="store_true",
        help="Launch Isaac Sim with GUI for screenshots and demos.",
    )
    parser.add_argument(
        "--keep-open",
        action="store_true",
        help="Keep the Isaac Sim window open after dataset generation.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Remove the existing synthetic dataset directory before generation.",
    )
    return parser.parse_args()


def build_gui_preview_scene() -> None:
    """Create a stable, visible scene for GUI screenshots."""
    import omni.usd  # type: ignore
    from pxr import Gf, Sdf, UsdGeom, UsdLux, UsdShade  # type: ignore

    stage = omni.usd.get_context().get_stage()
    root = "/World/Preview"

    UsdGeom.Xform.Define(stage, root)

    dome_light = UsdLux.DomeLight.Define(stage, f"{root}/DomeLight")
    dome_light.CreateIntensityAttr(5000)

    key_light = UsdLux.SphereLight.Define(stage, f"{root}/KeyLight")
    key_light.CreateIntensityAttr(12000)
    key_light.CreateRadiusAttr(1.0)
    key_light.AddTranslateOp().Set(Gf.Vec3d(0, -2.0, 4.0))

    fill_light = UsdLux.SphereLight.Define(stage, f"{root}/FillLight")
    fill_light.CreateIntensityAttr(6000)
    fill_light.CreateRadiusAttr(2.0)
    fill_light.AddTranslateOp().Set(Gf.Vec3d(-2.5, -1.0, 2.2))

    def create_preview_material(
        name: str,
        color: tuple[float, float, float],
        roughness: float = 0.45,
    ) -> UsdShade.Material:
        material = UsdShade.Material.Define(stage, f"{root}/Materials/{name}")
        shader = UsdShade.Shader.Define(stage, f"{root}/Materials/{name}/PreviewSurface")
        shader.CreateIdAttr("UsdPreviewSurface")
        shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color))
        shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(roughness)
        shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)
        material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
        return material

    def add_cube(
        name: str,
        position: tuple[float, float, float],
        scale: tuple[float, float, float],
        color: tuple[float, float, float],
    ) -> None:
        cube = UsdGeom.Cube.Define(stage, f"{root}/{name}")
        cube.CreateSizeAttr(1.0)
        cube.CreateDisplayColorAttr([Gf.Vec3f(*color)])
        cube.AddTranslateOp().Set(Gf.Vec3d(*position))
        cube.AddScaleOp().Set(Gf.Vec3f(*scale))

    def add_sphere(
        name: str,
        position: tuple[float, float, float],
        radius: float,
        color: tuple[float, float, float],
    ) -> None:
        sphere = UsdGeom.Sphere.Define(stage, f"{root}/{name}")
        sphere.CreateRadiusAttr(radius)
        sphere.CreateDisplayColorAttr([Gf.Vec3f(*color)])
        sphere.AddTranslateOp().Set(Gf.Vec3d(*position))

    def add_cylinder(
        name: str,
        position: tuple[float, float, float],
        radius: float,
        height: float,
        color: tuple[float, float, float],
    ) -> None:
        cylinder = UsdGeom.Cylinder.Define(stage, f"{root}/{name}")
        cylinder.CreateAxisAttr("Z")
        cylinder.CreateRadiusAttr(radius)
        cylinder.CreateHeightAttr(height)
        cylinder.CreateDisplayColorAttr([Gf.Vec3f(*color)])
        cylinder.AddTranslateOp().Set(Gf.Vec3d(*position))

    def add_cone(
        name: str,
        position: tuple[float, float, float],
        radius: float,
        height: float,
        color: tuple[float, float, float],
    ) -> None:
        cone = UsdGeom.Cone.Define(stage, f"{root}/{name}")
        cone.CreateAxisAttr("Z")
        cone.CreateRadiusAttr(radius)
        cone.CreateHeightAttr(height)
        cone.CreateDisplayColorAttr([Gf.Vec3f(*color)])
        cone.AddTranslateOp().Set(Gf.Vec3d(*position))

    def add_capsule(
        name: str,
        position: tuple[float, float, float],
        radius: float,
        height: float,
        rotation: tuple[float, float, float],
        color: tuple[float, float, float],
    ) -> None:
        capsule = UsdGeom.Capsule.Define(stage, f"{root}/{name}")
        capsule.CreateAxisAttr("Z")
        capsule.CreateRadiusAttr(radius)
        capsule.CreateHeightAttr(height)
        capsule.CreateDisplayColorAttr([Gf.Vec3f(*color)])
        capsule.AddTranslateOp().Set(Gf.Vec3d(*position))
        capsule.AddRotateXYZOp().Set(Gf.Vec3f(*rotation))

    def get_isaac_assets_root() -> str | None:
        try:
            from isaacsim.storage.native import get_assets_root_path  # type: ignore

            return get_assets_root_path()
        except Exception:
            try:
                from omni.isaac.core.utils.nucleus import get_assets_root_path  # type: ignore

                return get_assets_root_path()
            except Exception:
                return None

    white_robot_material = create_preview_material("WhiteRobot", (0.88, 0.9, 0.92), 0.35)

    def add_unitree_g1() -> bool:
        asset_path = os.getenv("HUMANOID_USD_PATH")
        if asset_path and asset_path.startswith("omniverse://localhost/"):
            print(
                "Ignoring HUMANOID_USD_PATH on omniverse://localhost because it often "
                "causes missing references when local Nucleus assets are not mounted."
            )
            asset_path = None

        if not asset_path:
            assets_root = get_isaac_assets_root()
            if assets_root:
                asset_path = f"{assets_root}/Isaac/Robots/Unitree/G1/g1.usd"

        if not asset_path:
            print("Unitree G1 asset root was not found. Falling back to primitive humanoid.")
            return False

        try:
            # Keep the referenced asset's own xform stack untouched. Some Isaac
            # robot USDs already define translate/orient/scale ops.
            g1_root_path = f"{root}/UnitreeG1Root"
            g1_asset_path = f"{g1_root_path}/UnitreeG1"
            g1_root = UsdGeom.Xform.Define(stage, g1_root_path)
            g1_root.AddTranslateOp().Set(Gf.Vec3d(-1.25, -0.25, 0.0))
            UsdShade.MaterialBindingAPI(g1_root.GetPrim()).Bind(white_robot_material)

            g1_prim = UsdGeom.Xform.Define(stage, g1_asset_path).GetPrim()
            if not g1_prim.GetReferences().AddReference(asset_path):
                print(f"Could not reference Unitree G1 asset: {asset_path}")
                return False
            UsdShade.MaterialBindingAPI(g1_prim).Bind(white_robot_material)

            print(f"Unitree G1 asset referenced from: {asset_path}")
            return True
        except Exception as exc:
            print(f"Could not load Unitree G1 asset. Falling back to primitive humanoid: {exc}")
            return False

    add_cube("Floor", (0, 0, -0.05), (5.0, 5.0, 0.05), (0.68, 0.68, 0.66))
    add_cube("Backdrop", (0, 2.05, 1.1), (5.0, 0.05, 1.2), (0.78, 0.8, 0.82))
    add_cone("TrafficCone", (-0.8, 0.35, 0.32), 0.22, 0.65, (1.0, 0.38, 0.02))
    add_cube("TrafficConeWhiteBand", (-0.8, 0.35, 0.33), (0.24, 0.24, 0.035), (0.96, 0.96, 0.9))
    add_cylinder("Barrel", (-0.25, 0.7, 0.36), 0.24, 0.72, (0.05, 0.26, 0.9))
    add_cube("Crate", (0.35, 0.45, 0.25), (0.32, 0.32, 0.25), (0.74, 0.43, 0.16))
    add_cube("Shelf", (0.95, 0.95, 0.55), (0.42, 0.18, 0.55), (0.55, 0.34, 0.16))
    add_cube("ShelfMiddleBoard", (0.95, 0.95, 0.55), (0.44, 0.2, 0.025), (0.28, 0.16, 0.07))

    if not add_unitree_g1():
        # Stylized humanoid robot mockup for project screenshots.
        robot_blue = (0.08, 0.25, 0.95)
        dark_blue = (0.03, 0.08, 0.28)
        joint_gray = (0.72, 0.76, 0.82)
        camera_cyan = (0.0, 0.95, 1.0)

        add_cube("HumanoidPelvis", (-1.25, -0.25, 0.88), (0.3, 0.14, 0.12), dark_blue)
        add_cube("HumanoidTorso", (-1.25, -0.25, 1.25), (0.28, 0.16, 0.34), robot_blue)
        add_cube("HumanoidChestPlate", (-1.25, -0.42, 1.32), (0.2, 0.035, 0.22), (0.0, 0.85, 1.0))
        add_capsule("HumanoidNeck", (-1.25, -0.25, 1.58), 0.045, 0.16, (0, 0, 0), joint_gray)
        add_sphere("HumanoidHead", (-1.25, -0.25, 1.76), 0.17, joint_gray)
        add_cube("HeadCameraMarker", (-1.25, -0.43, 1.76), (0.08, 0.035, 0.045), camera_cyan)

        add_sphere("LeftShoulderJoint", (-1.58, -0.25, 1.43), 0.07, joint_gray)
        add_sphere("RightShoulderJoint", (-0.92, -0.25, 1.43), 0.07, joint_gray)
        add_capsule("LeftUpperArm", (-1.66, -0.25, 1.22), 0.055, 0.36, (0, 18, 0), robot_blue)
        add_capsule("RightUpperArm", (-0.84, -0.25, 1.22), 0.055, 0.36, (0, -18, 0), robot_blue)
        add_sphere("LeftElbowJoint", (-1.72, -0.25, 1.02), 0.06, joint_gray)
        add_sphere("RightElbowJoint", (-0.78, -0.25, 1.02), 0.06, joint_gray)
        add_capsule("LeftForearm", (-1.72, -0.25, 0.84), 0.045, 0.28, (0, 0, 0), dark_blue)
        add_capsule("RightForearm", (-0.78, -0.25, 0.84), 0.045, 0.28, (0, 0, 0), dark_blue)
        add_cube("LeftGripper", (-1.72, -0.38, 0.67), (0.09, 0.035, 0.035), camera_cyan)
        add_cube("RightGripper", (-0.78, -0.38, 0.67), (0.09, 0.035, 0.035), camera_cyan)

        add_sphere("LeftHipJoint", (-1.4, -0.25, 0.78), 0.065, joint_gray)
        add_sphere("RightHipJoint", (-1.1, -0.25, 0.78), 0.065, joint_gray)
        add_capsule("LeftUpperLeg", (-1.4, -0.25, 0.52), 0.06, 0.42, (0, 0, 0), robot_blue)
        add_capsule("RightUpperLeg", (-1.1, -0.25, 0.52), 0.06, 0.42, (0, 0, 0), robot_blue)
        add_sphere("LeftKneeJoint", (-1.4, -0.25, 0.28), 0.06, joint_gray)
        add_sphere("RightKneeJoint", (-1.1, -0.25, 0.28), 0.06, joint_gray)
        add_capsule("LeftLowerLeg", (-1.4, -0.25, 0.05), 0.055, 0.36, (0, 0, 0), dark_blue)
        add_capsule("RightLowerLeg", (-1.1, -0.25, 0.05), 0.055, 0.36, (0, 0, 0), dark_blue)
        add_cube("LeftFoot", (-1.4, -0.42, -0.15), (0.12, 0.22, 0.04), joint_gray)
        add_cube("RightFoot", (-1.1, -0.42, -0.15), (0.12, 0.22, 0.04), joint_gray)

    camera = UsdGeom.Camera.Define(stage, f"{root}/HumanoidHeadCamera")
    camera.AddTranslateOp().Set(Gf.Vec3d(-1.25, -0.46, 1.62))
    camera.AddRotateXYZOp().Set(Gf.Vec3f(78, 0, 0))
    camera.CreateFocalLengthAttr(20)

    print(f"GUI preview scene created at {root}. Use Stage > World > Preview and press F if needed.")


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    log(f"Loaded config: {args.config}")

    # Isaac Sim modules must be imported after launching through python.sh.
    from omni.isaac.kit import SimulationApp  # type: ignore

    log(f"Launching Isaac Sim. gui={args.gui}, keep_open={args.keep_open}")
    sim_app = SimulationApp({"headless": not args.gui})

    import omni.replicator.core as rep  # type: ignore

    random.seed(config["scene"]["random_seed"])

    final_output_dir = Path(config["dataset_root"]).resolve()
    output_dir = final_output_dir
    if args.overwrite:
        output_dir = final_output_dir.with_name(f"{final_output_dir.name}_tmp")
        if output_dir.exists():
            log(f"Removing stale temporary output directory: {output_dir}")
            shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    camera_config = config["randomization"]["camera"]
    log(
        "Generating "
        f"{config['scene']['num_frames']} frames at "
        f"{config['scene']['image_width']}x{config['scene']['image_height']} "
        f"into {output_dir}"
    )

    with rep.new_layer():
        lighting_config = config["randomization"]["lighting"]
        light_min, light_max = lighting_config["intensity_range"]
        key_light_intensity = float(light_max if lighting_config["enabled"] else 60000)
        fill_light_intensity = float((light_min + light_max) / 2 if lighting_config["enabled"] else 25000)

        floor_material = create_visible_material(
            rep,
            color=rgb255(160, 160, 150),
            roughness=0.35,
        )
        backdrop_material = create_visible_material(
            rep,
            color=rgb255(185, 190, 190),
            roughness=0.55,
        )
        object_materials = {
            "traffic_cone": create_visible_material(rep, rgb255(255, 95, 20), 0.35),
            "barrel": create_visible_material(rep, rgb255(20, 80, 220), 0.3),
            "crate": create_visible_material(rep, rgb255(190, 110, 40), 0.45),
            "shelf": create_visible_material(rep, rgb255(95, 95, 92), 0.45),
        }

        camera = rep.create.camera(
            position=(0, -2.5, 1.55),
            rotation=(75, 0, 0),
            focal_length=20,
        )

        render_product = rep.create.render_product(
            camera,
            (config["scene"]["image_width"], config["scene"]["image_height"]),
        )

        rep.create.light(
            light_type="Dome",
            intensity=9000,
        )
        rep.create.light(
            light_type="Sphere",
            position=(0, -2.0, 4.0),
            scale=4.0,
            intensity=key_light_intensity,
        )
        rep.create.light(
            light_type="Sphere",
            position=(-2.5, -1.0, 2.4),
            scale=3.0,
            intensity=fill_light_intensity,
        )

        floor = rep.create.plane(
            scale=(8, 8, 1),
            position=(0, 0, 0),
            semantics=[("class", "floor")],
        )
        with floor:
            rep.randomizer.materials([floor_material])

        backdrop = rep.create.cube(
            scale=(5.0, 0.08, 1.5),
            position=(0, 1.55, 0.75),
            semantics=[("class", "background_wall")],
        )
        with backdrop:
            rep.randomizer.materials([backdrop_material])

        object_specs = {
            "traffic_cone": {
                "shape": "cone",
                "scale": (0.32, 0.32, 0.68),
                "z_range": (0.32, 0.38),
            },
            "barrel": {
                "shape": "cylinder",
                "scale": (0.34, 0.34, 0.72),
                "z_range": (0.34, 0.42),
            },
            "crate": {
                "shape": "cube",
                "scale": (0.42, 0.42, 0.34),
                "z_range": (0.18, 0.26),
            },
            "shelf": {
                "shape": "cube",
                "scale": (0.58, 0.22, 0.78),
                "z_range": (0.40, 0.52),
            },
        }

        object_prims = []
        for class_name in config["classes"]:
            object_spec = object_specs[class_name]
            create_primitive = getattr(rep.create, object_spec["shape"])
            primitive = create_primitive(
                scale=object_spec["scale"],
                position=(0, 0.6, object_spec["z_range"][0]),
                semantics=[("class", class_name)],
            )
            with primitive:
                rep.randomizer.materials([object_materials[class_name]])
            object_prims.append((primitive, object_spec))

        with rep.trigger.on_frame(num_frames=config["scene"]["num_frames"]):
            with camera:
                rep.modify.pose(
                    position=rep.distribution.uniform(
                        (
                            camera_config["lateral_range"][0],
                            -3.3,
                            0.9,
                        ),
                        (
                            camera_config["lateral_range"][1],
                            -1.6,
                            1.55,
                        ),
                    ),
                    look_at=rep.distribution.uniform((-0.7, 0.4, 0.25), (0.7, 1.2, 0.65)),
                )

            for primitive, object_spec in object_prims:
                with primitive:
                    rep.modify.pose(
                        position=rep.distribution.uniform(
                            (-0.9, 0.25, object_spec["z_range"][0]),
                            (0.9, 1.15, object_spec["z_range"][1]),
                        ),
                        rotation=rep.distribution.uniform((0, 0, 0), (0, 0, 360)),
                    )

        writer = rep.WriterRegistry.get("BasicWriter")
        writer.initialize(
            output_dir=str(output_dir),
            rgb=config["writer"]["rgb"],
            bounding_box_2d_tight=config["writer"]["bounding_box_2d_tight"],
            semantic_segmentation=config["writer"]["semantic_segmentation"],
            distance_to_camera=config["writer"]["depth"],
        )
        writer.attach([render_product])

        log("Warming up renderer before capture.")
        for _ in range(30):
            sim_app.update()

        log("Replicator writer attached. Starting orchestrator.")
        rep.orchestrator.run()
        rep.orchestrator.wait_until_complete()
        log("Replicator orchestrator completed.")

    generated_files = list(output_dir.glob("*"))
    rgb_files = list(output_dir.glob("rgb*"))
    json_files = list(output_dir.glob("*.json"))
    if args.overwrite:
        if not rgb_files:
            raise RuntimeError(
                "Synthetic data generation produced no RGB files. "
                f"Keeping existing output directory untouched: {final_output_dir}"
            )
        if final_output_dir.exists():
            log(f"Replacing existing output directory: {final_output_dir}")
            shutil.rmtree(final_output_dir)
        shutil.move(str(output_dir), str(final_output_dir))
        output_dir = final_output_dir
        generated_files = list(output_dir.glob("*"))
        rgb_files = list(output_dir.glob("rgb*"))
        json_files = list(output_dir.glob("*.json"))

    log(f"Synthetic dataset output directory: {output_dir}")
    log(f"Generated files: total={len(generated_files)}, rgb={len(rgb_files)}, json={len(json_files)}")

    if args.gui:
        build_gui_preview_scene()
        for _ in range(30):
            sim_app.update()

    if args.keep_open:
        log("Isaac Sim GUI will stay open. Press Ctrl+C in the terminal to close it.")
        try:
            while sim_app.is_running():
                sim_app.update()
        except KeyboardInterrupt:
            log("Closing Isaac Sim GUI.")

    sim_app.close()
    log("Isaac Sim closed.")


if __name__ == "__main__":
    main()
