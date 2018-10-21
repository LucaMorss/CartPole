set_param('penddemoTCPIPstop','FastRestart','on')
for i=1:100000 
    sim('penddemoTCPIPstop')
end