// WindDrag -- gusty wind and body drag on the rig v2 x500 in Gazebo: the same wind as wind/x500_mjx.py.
//   mean   W10 at 10 m with a log profile (z0 0.3 m, suburb) toward <direction_deg> (ENU, 0 = blowing east)
//   turbulence  MIL-F-8785C low altitude, first-order filters per axis (along, across, vertical)
//   gusts  1-cosine, Poisson 1/15 s, amplitude 0.2-0.6 W10, 1.5-5 s, within +-0.5 rad of the mean direction
// Each physics step it sets the world's wind (Gazebo's MulticopterMotorModel takes its propeller drag against it)
// and pushes the drone's base_link with body drag 0.5 rho CdA |v| v, which Gazebo's x500 does not have.
// It publishes the wind it made on /wind_truth (x, y, z m/s) for the logs. With <start_on>0 the air stays still until a
// message on /wind_scale (Double, 1 = on): the wind then starts from its seed, so every flight of an episode meets
// the same gusts from the moment its landing starts (the flight out from the depot is calm).
#include <atomic>
#include <cmath>
#include <random>
#include <string>

#include <gz/msgs/double.pb.h>
#include <gz/msgs/vector3d.pb.h>
#include <gz/plugin/Register.hh>
#include <gz/sim/Link.hh>
#include <gz/sim/Model.hh>
#include <gz/sim/System.hh>
#include <gz/sim/Util.hh>
#include <gz/sim/components/LinearVelocity.hh>
#include <gz/sim/components/Name.hh>
#include <gz/sim/components/Model.hh>
#include <gz/sim/components/Wind.hh>
#include <gz/transport/Node.hh>

using namespace gz;
using namespace sim;

class WindDrag : public System, public ISystemConfigure, public ISystemPreUpdate
{
  std::string model_name_ = "x500_mono_cam_down_0";
  double w10_ = 0, dir_ = 0, cda_h_ = 0.04, cda_v_ = 0.05, rho_ = 1.225, z0_ = 0.3, gust_rate_ = 1.0 / 15.0;
  double u_ = 0, v_ = 0, w_ = 0, gust_t_ = -1, gust_T_ = 0, gust_A_ = 0, gust_dir_ = 0, t_ = 0, pub_t_ = 0;
  bool turb_ = true;
  unsigned seed_ = 0;
  std::atomic<double> scale_{1.0};
  double applied_ = -1.0;
  std::mt19937 rng_{0};
  std::normal_distribution<double> n01_{0.0, 1.0};
  std::uniform_real_distribution<double> u01_{0.0, 1.0};
  Link link_{kNullEntity};
  transport::Node node_;
  transport::Node::Publisher pub_;

 public:
  void Configure(const Entity &, const std::shared_ptr<const sdf::Element> &sdf, EntityComponentManager &,
                 EventManager &) override
  {
    auto get = [&](const char *k, double d) { return sdf->HasElement(k) ? sdf->Get<double>(k) : d; };
    w10_ = get("mean_speed", 0.0);
    dir_ = get("direction_deg", 0.0) * M_PI / 180.0;
    cda_h_ = get("cda_h", cda_h_);
    cda_v_ = get("cda_v", cda_v_);
    z0_ = get("z0", z0_);
    gust_rate_ = get("gust_rate", gust_rate_);
    turb_ = get("turbulence", 1.0) > 0.5;
    seed_ = static_cast<unsigned>(get("seed", 0.0));
    rng_.seed(seed_);
    scale_ = get("start_on", 1.0);
    node_.Subscribe("/wind_scale", &WindDrag::OnScale, this);
    if (sdf->HasElement("model_name")) model_name_ = sdf->Get<std::string>("model_name");
    pub_ = node_.Advertise<msgs::Vector3d>("/wind_truth");
  }

  void OnScale(const msgs::Double &m) { scale_ = m.data(); }

  void PreUpdate(const UpdateInfo &info, EntityComponentManager &ecm) override
  {
    if (info.paused) return;
    const double sc = scale_;
    if (sc > 0.5 && applied_ <= 0.5) {                     // switched on: the same wind from its seed, every time
      rng_.seed(seed_); u_ = v_ = w_ = 0; gust_t_ = -1; t_ = 0;
    }
    applied_ = sc;
    const double dt = std::chrono::duration<double>(info.dt).count();
    if (dt <= 0) return;
    t_ += dt;
    if (link_.Entity() == kNullEntity) {
      auto m = ecm.EntityByComponents(components::Model(), components::Name(model_name_));
      if (m == kNullEntity) return;
      link_ = Link(Model(m).LinkByName(ecm, "base_link"));
      link_.EnableVelocityChecks(ecm, true);
      return;
    }
    auto pose = link_.WorldPose(ecm);
    auto vel = link_.WorldLinearVelocity(ecm);
    if (!pose || !vel) return;
    const double h = std::max(pose->Pos().Z(), 1.0);
    const double wm = w10_ * std::log(h / z0_) / std::log(10.0 / z0_);
    const math::Vector3d ea(std::cos(dir_), std::sin(dir_), 0), ec(-std::sin(dir_), std::cos(dir_), 0);
    if (turb_ && w10_ > 0) {
      const double hf = std::max(h, 3.0) * 3.281, den = 0.177 + 0.000823 * hf;
      const double sw = 0.1 * w10_, su = sw / std::pow(den, 0.4);
      const double lu = hf / std::pow(den, 1.2) / 3.281, lw = std::max(h, 3.0), V = std::max(wm, 1.0);
      const double au = std::exp(-V * dt / lu), aw = std::exp(-V * dt / lw);
      u_ = au * u_ + su * std::sqrt(1 - au * au) * n01_(rng_);
      v_ = au * v_ + su * std::sqrt(1 - au * au) * n01_(rng_);
      w_ = aw * w_ + sw * std::sqrt(1 - aw * aw) * n01_(rng_);
    }
    math::Vector3d gust(0, 0, 0);
    if (gust_t_ < 0 && w10_ > 0 && u01_(rng_) < gust_rate_ * dt) {      // a gust starts
      gust_t_ = 0; gust_T_ = 1.5 + 3.5 * u01_(rng_); gust_A_ = w10_ * (0.2 + 0.4 * u01_(rng_));
      gust_dir_ = dir_ + (u01_(rng_) - 0.5);
    }
    if (gust_t_ >= 0) {
      const double a = gust_A_ * 0.5 * (1 - std::cos(2 * M_PI * gust_t_ / gust_T_));
      gust = math::Vector3d(std::cos(gust_dir_), std::sin(gust_dir_), 0) * a;
      gust_t_ += dt;
      if (gust_t_ > gust_T_) gust_t_ = -1;
    }
    const math::Vector3d wind = (ea * (wm + u_) + ec * v_ + math::Vector3d(0, 0, w_) + gust) * sc;
    auto went = ecm.EntityByComponents(components::Wind());
    if (went != kNullEntity) {
      if (!ecm.Component<components::WorldLinearVelocity>(went))
        ecm.CreateComponent(went, components::WorldLinearVelocity(wind));
      else
        ecm.SetComponentData<components::WorldLinearVelocity>(went, wind);
    }
    const math::Vector3d vr = *vel - wind;
    const double vh = std::hypot(vr.X(), vr.Y());
    link_.AddWorldForce(ecm, math::Vector3d(-0.5 * rho_ * cda_h_ * vh * vr.X(), -0.5 * rho_ * cda_h_ * vh * vr.Y(),
                                            -0.5 * rho_ * cda_v_ * std::abs(vr.Z()) * vr.Z()));
    if (t_ - pub_t_ >= 0.02) {
      pub_t_ = t_;
      msgs::Vector3d m;
      m.set_x(wind.X()); m.set_y(wind.Y()); m.set_z(wind.Z());
      pub_.Publish(m);
    }
  }
};

GZ_ADD_PLUGIN(WindDrag, System, WindDrag::ISystemConfigure, WindDrag::ISystemPreUpdate)
GZ_ADD_PLUGIN_ALIAS(WindDrag, "WindDrag")
