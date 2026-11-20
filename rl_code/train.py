from rl_code.channel_gen import ChannelGenerator

def main():
    cir_generator = ChannelGenerator()
    h, tau = cir_generator.generate_rayleigh_cir()
    print(h)
    print(tau)

if __name__ == "__main__":
    main()