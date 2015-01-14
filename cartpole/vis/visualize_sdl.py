#!/usr/bin/env python
import pygame
import math

class visualize_sdl(object):
    def init_vis(self,display_width,display_height,axis_x_min,axis_x_max,axis_y_min,axis_y_max,fps):
        pygame.init()
        self.screen = pygame.display.set_mode((display_width,display_height))
        pygame.display.set_caption("Cart Pole Display")

        self.axis = [axis_x_min,axis_x_max,axis_y_min,axis_y_max]
        self.display_width = display_width
        self.display_height = display_height

        pygame.display.flip()
        self.clock = pygame.time.Clock()
        self.delay = fps
        self.framenum = 0

    #call this every iteration to slow down to real time
    def delay_vis(self):
        self.clock.tick(self.delay)

    #call this to update the visualization event loop
    #return True if the user wants to exit. return False otherwise
    def update_vis(self):

#        self.cp_sim.step()
#        if(self.cp_sim.state[2] < -4.0):
#            self.cp_sim.state[2] = -4.0
#        if(self.cp_sim.state[2] > 4.0):
#            self.cp_sim.state[2] = 4.0
                
#        self.cp_sim.u = 0.0
#        if(pressed[pygame.K_LEFT]):
#            self.cp_sim.u = -10.0
#        if(pressed[pygame.K_RIGHT]):
#            self.cp_sim.u = 10.0
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return True;
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                return True;
        return False

    def get_keys(self):
        keys = [];
        pressed = pygame.key.get_pressed()
        keys.append(pressed[pygame.K_LEFT])
        keys.append(pressed[pygame.K_RIGHT])
        keys.append(pressed[pygame.K_UP])
        keys.append(pressed[pygame.K_DOWN])
        return keys

    #convert x,y coords from the axis scale to screen coordinates
    def convert_coords(self,coords):
        img_coords = [0,0]
        img_width = self.display_width
        img_height = self.display_height
        img_coords[0] = float(((coords[0] - self.axis[0])/float(self.axis[1] - self.axis[0]))*float(img_width))
        img_coords[1] = float(((coords[1] - self.axis[2])/float(self.axis[3] - self.axis[2]))*float(img_height))
        return tuple(img_coords);

    #convert x,y,width,height coords from the axis scale to screen coordinates
    def convert_rect_coords(self,coords):
        img_coords = [0,0,0,0]
        img_width = self.display_width
        img_height = self.display_height
        img_coords[0] = float(((coords[0] - self.axis[0])/float(self.axis[1] - self.axis[0]))*float(img_width))
        img_coords[1] = float(((coords[1] - self.axis[2])/float(self.axis[3] - self.axis[2]))*float(img_height))
        img_coords[2] = float(((coords[2])/float(self.axis[1] - self.axis[0]))*float(img_width))
        img_coords[3] = float(((coords[3])/float(self.axis[3] - self.axis[2]))*float(img_height))
        return tuple(img_coords);

    #call this before calling update vis
    def draw_cartpole(self,state,action,reward,stats=None):
        #draw cart
        self.screen.fill((0,0,0))
        angle = state[0]
        pos = state[2]
        coords = self.convert_rect_coords((pos - 0.5,-0.5,1.0,1.0))
        if(reward > 0.001):
            color = (255,96,96)
        else:
            color = (192,24,24)
        self.screen.fill(color,pygame.Rect(coords))
        
        #draw pole
        pole_coords1 = self.convert_coords((pos,0.0))
        pole_coords2 = self.convert_coords((pos + 4.0*math.sin(angle),-4.0*math.cos(angle)))
        pygame.draw.line(self.screen,(24,24,192),pole_coords1,pole_coords2,4)

        #Draw arrow from action
        #"-" and "/" and "\" strokes makes "->"
        if(action == 1):
            # "-" stroke of arrow
            arrow_coords1 = self.convert_coords((pos+0.25,0.0))
            arrow_coords2 = self.convert_coords((pos+0.75,0.0))
            pygame.draw.line(self.screen,(224,224,224),arrow_coords1,arrow_coords2,1)

            # "\" stroke of arrow
            arrow_coords1 = self.convert_coords((pos+0.75,0.0))
            arrow_coords2 = self.convert_coords((pos+0.50,-0.25))
            pygame.draw.line(self.screen,(224,224,224),arrow_coords1,arrow_coords2,1)

            # "/" stroke of arrow
            arrow_coords1 = self.convert_coords((pos+0.75,0.0))
            arrow_coords2 = self.convert_coords((pos+0.50,+0.25))
            pygame.draw.line(self.screen,(224,224,224),arrow_coords1,arrow_coords2,1)

        if(action == 2):
            # "-" stroke of arrow
            arrow_coords1 = self.convert_coords((pos-0.25,0.0))
            arrow_coords2 = self.convert_coords((pos-0.75,0.0))
            pygame.draw.line(self.screen,(224,224,224),arrow_coords1,arrow_coords2,1)

            # "\" stroke of arrow
            arrow_coords1 = self.convert_coords((pos-0.75,0.0))
            arrow_coords2 = self.convert_coords((pos-0.50,-0.25))
            pygame.draw.line(self.screen,(224,224,224),arrow_coords1,arrow_coords2,1)

            # "/" stroke of arrow
            arrow_coords1 = self.convert_coords((pos-0.75,0.0))
            arrow_coords2 = self.convert_coords((pos-0.50,+0.25))
            pygame.draw.line(self.screen,(224,224,224),arrow_coords1,arrow_coords2,1)
        if(stats is not None):    
            line_pos=30
            font = pygame.font.SysFont("Ubuntu Mono",20)
            height = font.get_height()*1.2

            text = font.render(("Episode: " + str(stats.episode)),1,(255,255,255))
            self.screen.blit(text,(30,line_pos))

            line_pos += height
            text = font.render(("Step: " + str(stats.step)),1,(255,255,255))
            self.screen.blit(text,(30,line_pos))

            line_pos += height
            text = font.render(("Reward: " + str(stats.r)),1,(255,255,255))
            self.screen.blit(text,(30,line_pos))

            line_pos += height
            text = font.render(("Total Reward This Episode: " + str(stats.r_sum)),1,(255,255,255))
            self.screen.blit(text,(30,line_pos))

            line_pos += height
            text = font.render(("Average Reward Per Episode: " + str(stats.r_sum_avg)),1,(255,255,255))
            self.screen.blit(text,(30,line_pos))
            
            line_pos += height
            text = font.render(("Current Epsilon: " + str(stats.epsilon)),1,(255,255,255))
            self.screen.blit(text,(30,line_pos))

            line_pos += height
            text = font.render(("Current Gamma: " + str(stats.gamma)),1,(255,255,255))
            self.screen.blit(text,(30,line_pos))

            line_pos += height
            text = font.render(("Current Alpha: " + str(stats.alpha)),1,(255,255,255))
            self.screen.blit(text,(30,line_pos))
        if(stats.fast_forward):
            line_pos=30
            font = pygame.font.SysFont("Ubuntu",100)

            text = font.render(("FAST FORWARD"),1,(128,255,255))
            self.screen.blit(text,(300,300))

        if(stats.save_images):
            pygame.image.save(self.screen,stats.image_save_dir + "frame_" + str(self.framenum) + ".png")
            self.framenum = self.framenum + 1

            
        pygame.display.flip()


#this runs a simple keyboard driven test, with no simulator for the cart-pole
if __name__ == '__main__':
    v = visualize_sdl()
    v.init_vis()
    state = [0.0,0.0,0.0,0.0]
    while 1:
        v.delay_vis()
        k = v.get_keys()
        if(k[0]):
            state[0] -= 0.1
        if(k[1]):
            state[0] += 0.1
        if(k[2]):
            state[2] -= 0.1
        if(k[3]):
            state[2] += 0.1
        v.draw_cartpole(state)
        exit = v.update_vis()
        if(exit):
            break

